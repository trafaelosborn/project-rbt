"""
Tests for cosine acceleration layer:
  1. _sparse_profile_cosine correctness (union→intersection fix)
  2. LatinFormReference.score_token fast path
  3. FortranCosineScorer (Fortran + numpy fallback)
  4. IncrementalScoringState with Fortran scorer vs Python path
"""
from __future__ import annotations

import math
from collections import Counter

import numpy as np
import pytest

from src.retrodiction.engine_reinforced_v2 import (
    LatinFormReference,
    _sparse_profile_cosine,
    _build_sparse_profile,
    _extract_char_ngrams_from_sequences,
    _extract_suffixes_from_sequences,
    CHAR_BIGRAM_TOP_N,
    CHAR_TRIGRAM_TOP_N,
    SUFFIX_TOP_N,
    SUFFIX_LEN,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _old_sparse_profile_cosine(a: dict, b: dict) -> float:
    """Original (slow) implementation using set union — reference for comparison."""
    if not a or not b:
        return 0.0
    dot = sum(a.get(k, 0.0) * b.get(k, 0.0) for k in set(a) | set(b))
    na = sum(v * v for v in a.values()) ** 0.5
    nb = sum(v * v for v in b.values()) ** 0.5
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(dot / (na * nb))


# ---------------------------------------------------------------------------
# _sparse_profile_cosine
# ---------------------------------------------------------------------------

class TestSparsProfileCosine:
    def test_identical_profiles(self):
        a = {"x": 0.5, "y": 0.3, "z": 0.2}
        assert abs(_sparse_profile_cosine(a, a) - 1.0) < 1e-9

    def test_orthogonal_profiles(self):
        a = {"x": 1.0}
        b = {"y": 1.0}
        assert _sparse_profile_cosine(a, b) == pytest.approx(0.0)

    def test_partial_overlap(self):
        a = {"an": 0.5, "am": 0.3, "or": 0.2}
        b = {"an": 0.6, "or": 0.4, "es": 0.3}
        new = _sparse_profile_cosine(a, b)
        old = _old_sparse_profile_cosine(a, b)
        assert new == pytest.approx(old, rel=1e-8), \
            f"Fixed version {new!r} must match old version {old!r}"

    def test_small_a_large_b(self):
        """Key case: small candidate profile vs large Latin reference profile."""
        a = {"am": 0.5, "^a": 0.3, "r$": 0.2}
        # Simulate a Latin reference with many entries
        b = {f"k{i}": 0.01 for i in range(800)}
        b.update({"am": 0.08, "^a": 0.05})
        new = _sparse_profile_cosine(a, b)
        old = _old_sparse_profile_cosine(a, b)
        assert new == pytest.approx(old, rel=1e-8)

    def test_empty_a(self):
        assert _sparse_profile_cosine({}, {"x": 1.0}) == 0.0

    def test_empty_b(self):
        assert _sparse_profile_cosine({"x": 1.0}, {}) == 0.0

    def test_no_overlap(self):
        a = {"aa": 1.0}
        b = {"bb": 1.0}
        assert _sparse_profile_cosine(a, b) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# LatinFormReference.score_token fast path
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def latin_ref():
    return LatinFormReference()


class TestScoreTokenFastPath:
    def test_result_in_zero_one(self, latin_ref):
        for tok in ["amor", "est", "vita", "corpus", "ab", "xyzabc", ""]:
            score = latin_ref.score_token(tok)
            assert 0.0 <= score <= 1.0, f"score_token({tok!r}) = {score} out of [0,1]"

    def test_empty_token(self, latin_ref):
        assert latin_ref.score_token("") == 0.0

    def test_matches_legacy_score_method(self, latin_ref):
        """Fast path must match the full score() path to floating-point precision."""
        test_tokens = ["amor", "vita", "corpus", "est", "xyzabc", "amicitia", "in", "se"]
        for tok in test_tokens:
            # Force recompute by clearing cache
            latin_ref._score_token_cache.pop(tok, None)
            fast = latin_ref.score_token(tok)
            slow = latin_ref.score([[tok]])["latin_form_score"]
            assert abs(fast - slow) < 1e-6, \
                f"score_token({tok!r}): fast={fast:.8f} slow={slow:.8f}"

    def test_latin_like_scores_higher_than_random(self, latin_ref):
        latin = latin_ref.score_token("amicitia")
        random = latin_ref.score_token("xzqwkp")
        assert latin > random

    def test_cache_returns_same_value(self, latin_ref):
        tok = "fiducia"
        latin_ref._score_token_cache.pop(tok, None)
        v1 = latin_ref.score_token(tok)
        v2 = latin_ref.score_token(tok)  # should hit cache
        assert v1 == v2

    def test_custom_form_weights_are_normalized_and_used(self):
        latin_ref = LatinFormReference(
            char_bigram_weight=1.0,
            char_trigram_weight=1.0,
            suffix_weight=0.0,
        )
        assert latin_ref.char_bigram_weight == pytest.approx(0.5)
        assert latin_ref.char_trigram_weight == pytest.approx(0.5)
        assert latin_ref.suffix_weight == pytest.approx(0.0)

        seqs = [["amor", "est", "vita"] * 10]
        bg_prof = _build_sparse_profile(_extract_char_ngrams_from_sequences(seqs, 2), CHAR_BIGRAM_TOP_N)
        tg_prof = _build_sparse_profile(_extract_char_ngrams_from_sequences(seqs, 3), CHAR_TRIGRAM_TOP_N)
        sfx_prof = _build_sparse_profile(_extract_suffixes_from_sequences(seqs, SUFFIX_LEN), SUFFIX_TOP_N)
        expected = (
            0.5 * _sparse_profile_cosine(bg_prof, latin_ref.char_bigram_profile)
            + 0.5 * _sparse_profile_cosine(tg_prof, latin_ref.char_trigram_profile)
            + 0.0 * _sparse_profile_cosine(sfx_prof, latin_ref.suffix_profile)
        )
        observed = latin_ref.score(seqs)["latin_form_score"]
        assert observed == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# FortranCosineScorer
# ---------------------------------------------------------------------------

class TestFortranCosineScorer:
    @pytest.fixture(scope="class")
    def scorer(self, latin_ref):
        from src.accelerate.fortran_cosine import FortranCosineScorer
        return FortranCosineScorer.build(latin_ref)

    def test_using_fortran_or_numpy(self, scorer):
        # Either Fortran compiled or numpy fallback — both are valid
        assert isinstance(scorer.using_fortran, bool)

    def test_score_single_form_range(self, scorer):
        # All-zero counters → score 0 (no match)
        score = scorer.score_single_form(Counter(), Counter(), Counter())
        assert score == pytest.approx(0.0)

    def test_score_single_form_matches_python_path(self, scorer, latin_ref):
        seqs = [["amor", "est", "vita"] * 30]
        bg_c = _extract_char_ngrams_from_sequences(seqs, 2)
        tg_c = _extract_char_ngrams_from_sequences(seqs, 3)
        sfx_c = _extract_suffixes_from_sequences(seqs, SUFFIX_LEN)

        fast = scorer.score_single_form(
            bg_c, tg_c, sfx_c,
            bg_top_n=CHAR_BIGRAM_TOP_N,
            tg_top_n=CHAR_TRIGRAM_TOP_N,
            sfx_top_n=SUFFIX_TOP_N,
        )

        bg_prof  = _build_sparse_profile(bg_c, CHAR_BIGRAM_TOP_N)
        tg_prof  = _build_sparse_profile(tg_c, CHAR_TRIGRAM_TOP_N)
        sfx_prof = _build_sparse_profile(sfx_c, SUFFIX_TOP_N)
        py_score = (
            0.40 * _sparse_profile_cosine(bg_prof,  latin_ref.char_bigram_profile)
            + 0.40 * _sparse_profile_cosine(tg_prof,  latin_ref.char_trigram_profile)
            + 0.20 * _sparse_profile_cosine(sfx_prof, latin_ref.suffix_profile)
        )
        assert abs(fast - py_score) < 1e-3  # float32 tolerance

    def test_score_form_batch_matches_score_single(self, scorer):
        seqs = [["amor", "est"] * 20]
        bg_c = _extract_char_ngrams_from_sequences(seqs, 2)
        tg_c = _extract_char_ngrams_from_sequences(seqs, 3)
        sfx_c = _extract_suffixes_from_sequences(seqs, SUFFIX_LEN)

        single = scorer.score_single_form(bg_c, tg_c, sfx_c)
        batch  = scorer.score_form_batch([(bg_c, tg_c, sfx_c)])
        assert abs(single - float(batch[0])) < 1e-5

    def test_score_form_batch_from_deltas_matches_full_batch(self, scorer):
        base = [["amor", "est", "vita"] * 12]
        cand_a = [["nova", "est", "vita"] * 12]
        cand_b = [["amor", "erat", "vita"] * 12]

        def _delta(after: Counter, before: Counter) -> Counter:
            delta = Counter(after)
            delta.subtract(before)
            return delta

        base_bg = _extract_char_ngrams_from_sequences(base, 2)
        base_tg = _extract_char_ngrams_from_sequences(base, 3)
        base_sfx = _extract_suffixes_from_sequences(base, SUFFIX_LEN)

        cand_a_bg = _extract_char_ngrams_from_sequences(cand_a, 2)
        cand_a_tg = _extract_char_ngrams_from_sequences(cand_a, 3)
        cand_a_sfx = _extract_suffixes_from_sequences(cand_a, SUFFIX_LEN)

        cand_b_bg = _extract_char_ngrams_from_sequences(cand_b, 2)
        cand_b_tg = _extract_char_ngrams_from_sequences(cand_b, 3)
        cand_b_sfx = _extract_suffixes_from_sequences(cand_b, SUFFIX_LEN)

        full_batch = scorer.score_form_batch(
            [
                (cand_a_bg, cand_a_tg, cand_a_sfx),
                (cand_b_bg, cand_b_tg, cand_b_sfx),
            ]
        )
        delta_batch = scorer.score_form_batch_from_deltas(
            base_bg,
            base_tg,
            base_sfx,
            [
                (_delta(cand_a_bg, base_bg), _delta(cand_a_tg, base_tg), _delta(cand_a_sfx, base_sfx)),
                (_delta(cand_b_bg, base_bg), _delta(cand_b_tg, base_tg), _delta(cand_b_sfx, base_sfx)),
            ],
        )
        assert np.allclose(full_batch, delta_batch, atol=1e-5)

    def test_score_form_batch_empty(self, scorer):
        result = scorer.score_form_batch([])
        assert result.shape == (0,)

    def test_custom_weights_apply_in_batch_path(self):
        from src.accelerate.fortran_cosine import FortranCosineScorer

        latin_ref = LatinFormReference(
            char_bigram_weight=1.0,
            char_trigram_weight=1.0,
            suffix_weight=0.0,
        )
        scorer = FortranCosineScorer.build(latin_ref)

        seqs = [["amor", "est", "vita"] * 20]
        bg_c = _extract_char_ngrams_from_sequences(seqs, 2)
        tg_c = _extract_char_ngrams_from_sequences(seqs, 3)
        sfx_c = _extract_suffixes_from_sequences(seqs, SUFFIX_LEN)

        batch = scorer.score_form_batch([(bg_c, tg_c, sfx_c)])

        bg_prof = _build_sparse_profile(bg_c, CHAR_BIGRAM_TOP_N)
        tg_prof = _build_sparse_profile(tg_c, CHAR_TRIGRAM_TOP_N)
        expected = (
            0.5 * _sparse_profile_cosine(bg_prof, latin_ref.char_bigram_profile)
            + 0.5 * _sparse_profile_cosine(tg_prof, latin_ref.char_trigram_profile)
        )
        assert float(batch[0]) == pytest.approx(expected, rel=1e-3)


# ---------------------------------------------------------------------------
# IncrementalScoringState with Fortran scorer
# ---------------------------------------------------------------------------

class TestIncrementalWithFortranScorer:
    @pytest.fixture(scope="class")
    def setup(self, latin_ref):
        from src.retrodiction.engine_reinforced import LatinReference
        from src.retrodiction.similarity import ReferenceSet
        from src.accelerate.incremental_scoring_state import IncrementalScoringState
        from src.accelerate.fortran_cosine import FortranCosineScorer

        seqs = [["amor", "est", "vita", f"tok{i}"] for i in range(40)]
        latin_struct = LatinReference()
        refs = ReferenceSet()
        scorer = FortranCosineScorer.build(latin_ref)

        state_py   = IncrementalScoringState.from_sequences(seqs, latin_ref, latin_struct, refs)
        state_fort = IncrementalScoringState.from_sequences(seqs, latin_ref, latin_struct, refs,
                                                            fortran_cosine_scorer=scorer)
        return state_py, state_fort, seqs

    def test_evaluate_form_scores_close(self, setup):
        state_py, state_fort, seqs = setup
        new_seqs = [list(s) for s in seqs]
        new_seqs[0] = ["nova", "res", "est"]

        scores_py   = state_py.evaluate(new_seqs, 0.0, 0.75, 0.05, 0.005)
        scores_fort = state_fort.evaluate(new_seqs, 0.0, 0.75, 0.05, 0.005)

        # Form scores may differ slightly (float32 vs float64, normalization differs
        # for small corpora where the Fortran path skips most_common top-n).
        assert abs(scores_py.latin_form_score - scores_fort.latin_form_score) < 1e-2

    def test_total_scores_close(self, setup):
        state_py, state_fort, seqs = setup
        new_seqs = [list(s) for s in seqs]
        new_seqs[1] = ["aqua", "ignis"]

        scores_py   = state_py.evaluate(new_seqs, 0.1, 0.75, 0.05, 0.005)
        scores_fort = state_fort.evaluate(new_seqs, 0.1, 0.75, 0.05, 0.005)

        # Structural score should match exactly; form may differ by float32
        assert abs(scores_py.latin_structural_score - scores_fort.latin_structural_score) < 1e-8
        assert abs(scores_py.total_score - scores_fort.total_score) < 1e-2

    def test_evaluate_batch_scores_close(self, setup):
        state_py, state_fort, seqs = setup
        cand_a = [list(s) for s in seqs]
        cand_b = [list(s) for s in seqs]
        cand_a[0] = ["nova", "res", "est"]
        cand_b[1] = ["aqua", "ignis"]

        batch_py = state_py.evaluate_batch([cand_a, cand_b], [0.0, 0.1], 0.75, 0.05, 0.005)
        batch_fort = state_fort.evaluate_batch([cand_a, cand_b], [0.0, 0.1], 0.75, 0.05, 0.005)

        for py_score, fort_score in zip(batch_py, batch_fort):
            assert abs(py_score.latin_structural_score - fort_score.latin_structural_score) < 1e-8
            assert abs(py_score.total_score - fort_score.total_score) < 1e-2
