"""Tests for src.accelerate.semantic_transparency and batch_candidate_scorer."""

import pytest
from collections import Counter

from src.accelerate.semantic_transparency import (
    TRANSPARENCY_TOP_N,
    SemanticTransparencyScorer,
    TransparencyResult,
)


# ---------------------------------------------------------------------------
# Minimal fake LatinFormReference for scoring tests
# ---------------------------------------------------------------------------

class _FakeFormRef:
    """Score token by whether it ends in -um/-us/-is (Latin-like)."""

    def score_token(self, token: str) -> float:
        if not token:
            return 0.0
        if token.endswith(("um", "us", "is", "em", "ae")):
            return 0.8
        if token.endswith(("nt", "re", "at")):
            return 0.4
        return 0.1


# ---------------------------------------------------------------------------
# SemanticTransparencyScorer unit tests
# ---------------------------------------------------------------------------

class TestSemanticTransparencyScorer:

    def _sequences_high(self):
        """Corpus dominated by Latin-like tokens."""
        return [
            ["dominum", "bellum", "romanus", "legiones"],
            ["dominum", "bellum", "dictator", "forum"],
            ["legiones", "bellum", "dominum", "romanus"],
        ] * 10

    def _sequences_low(self):
        """Corpus dominated by non-Latin-looking filler."""
        return [
            ["xx", "yy", "zz", "qq"],
            ["xx", "yy", "zz", "pp"],
            ["xx", "yy", "qq", "rr"],
        ] * 10

    def test_score_returns_float_in_unit_interval(self):
        scorer = SemanticTransparencyScorer(_FakeFormRef())
        for seqs in [self._sequences_high(), self._sequences_low()]:
            s = scorer.score(seqs)
            assert isinstance(s, float)
            assert 0.0 <= s <= 1.0

    def test_high_latin_sequences_score_above_low(self):
        scorer = SemanticTransparencyScorer(_FakeFormRef())
        high = scorer.score(self._sequences_high())
        low = scorer.score(self._sequences_low())
        assert high > low, f"high={high:.3f} should exceed low={low:.3f}"

    def test_score_full_returns_result(self):
        scorer = SemanticTransparencyScorer(_FakeFormRef())
        result = scorer.score_full(self._sequences_high())
        assert isinstance(result, TransparencyResult)
        assert result.score >= 0.0
        assert result.top_n > 0
        assert len(result.token_breakdown) > 0

    def test_token_breakdown_has_correct_structure(self):
        scorer = SemanticTransparencyScorer(_FakeFormRef())
        result = scorer.score_full(self._sequences_high())
        for tok, freq, score in result.token_breakdown:
            assert isinstance(tok, str)
            assert 0.0 < freq <= 1.0
            assert 0.0 <= score <= 1.0

    def test_empty_corpus_returns_zero(self):
        scorer = SemanticTransparencyScorer(_FakeFormRef())
        result = scorer.score_full([])
        assert result.score == 0.0
        assert result.top_n == 0

    def test_top_n_respected(self):
        scorer = SemanticTransparencyScorer(_FakeFormRef(), top_n=5)
        result = scorer.score_full(self._sequences_high())
        assert result.top_n <= 5

    def test_high_score_freq_mass_correct_direction(self):
        """Latin-like sequences should have more high-scoring tokens."""
        scorer = SemanticTransparencyScorer(_FakeFormRef())
        high_result = scorer.score_full(self._sequences_high())
        low_result = scorer.score_full(self._sequences_low())
        assert high_result.high_score_freq_mass > low_result.high_score_freq_mass

    def test_score_consistent_with_score_full(self):
        scorer = SemanticTransparencyScorer(_FakeFormRef())
        seqs = self._sequences_high()
        assert scorer.score(seqs) == pytest.approx(scorer.score_full(seqs).score)

    def test_from_form_ref_constructor(self):
        scorer = SemanticTransparencyScorer.from_form_ref(_FakeFormRef(), top_n=20)
        assert scorer._top_n == 20
        s = scorer.score(self._sequences_high())
        assert 0.0 <= s <= 1.0

    def test_single_token_corpus(self):
        scorer = SemanticTransparencyScorer(_FakeFormRef())
        seqs = [["bellum"]] * 5
        result = scorer.score_full(seqs)
        assert result.score > 0.5  # "bellum" ends in -um -> 0.8

    def test_frequency_weighting_favours_common_tokens(self):
        """Replacing common filler with Latin tokens should raise the score."""
        scorer = SemanticTransparencyScorer(_FakeFormRef())
        # Many repetitions of filler plus rare Latin
        filler_heavy = [["xx", "xx", "xx", "bellum"]] * 20
        # Many repetitions of Latin plus rare filler
        latin_heavy = [["bellum", "bellum", "bellum", "xx"]] * 20
        assert scorer.score(latin_heavy) > scorer.score(filler_heavy)


# ---------------------------------------------------------------------------
# BatchCandidateScorer tests
# ---------------------------------------------------------------------------

class TestPythonBatchCandidateScorer:

    def _build_scorer(self):
        from src.retrodiction.engine_reinforced import LatinReference
        from src.retrodiction.engine_reinforced_v2 import LatinFormReference
        from src.retrodiction.similarity import ReferenceSet
        from src.accelerate.batch_candidate_scorer import PythonBatchCandidateScorer
        return PythonBatchCandidateScorer(
            latin_structural_ref=LatinReference(),
            latin_form_ref=LatinFormReference(),
            references=ReferenceSet(),
        )

    def _tiny_candidates(self, n: int = 3) -> tuple[list, list]:
        base = [["bellum", "forum", "romanum", "datum"]] * 24
        candidates = [base[:] for _ in range(n)]
        costs = [0.1 * (i + 1) for i in range(n)]
        return candidates, costs

    def test_returns_one_result_per_candidate(self):
        scorer = self._build_scorer()
        candidates, costs = self._tiny_candidates(4)
        results = scorer.score_batch(candidates, costs, 0.75, 0.05, 0.005)
        assert len(results) == 4

    def test_candidate_index_matches_input_order(self):
        scorer = self._build_scorer()
        candidates, costs = self._tiny_candidates(3)
        results = scorer.score_batch(candidates, costs, 0.75, 0.05, 0.005)
        for i, r in enumerate(results):
            assert r.candidate_index == i

    def test_scores_are_finite_floats(self):
        scorer = self._build_scorer()
        candidates, costs = self._tiny_candidates(2)
        results = scorer.score_batch(candidates, costs, 0.75, 0.05, 0.005)
        import math
        for r in results:
            assert math.isfinite(r.total_score)
            assert math.isfinite(r.latin_structural_score)
            assert math.isfinite(r.latin_form_score)

    def test_higher_cost_reduces_total_score(self):
        scorer = self._build_scorer()
        base = [["bellum", "forum", "romanum"]] * 24
        r_low = scorer.score_batch([base], [0.1], 0.75, 0.05, 0.1)[0]
        r_high = scorer.score_batch([base], [5.0], 0.75, 0.05, 0.1)[0]
        assert r_low.total_score > r_high.total_score

    def test_transparency_scorer_adds_to_total(self):
        scorer = self._build_scorer()
        t_scorer = SemanticTransparencyScorer(_FakeFormRef())
        base = [["bellum", "forum", "romanum"]] * 24
        r_no_t = scorer.score_batch([base], [0.1], 0.75, 0.05, 0.005,
                                    transparency_scorer=None, transparency_weight=0.0)[0]
        r_with_t = scorer.score_batch([base], [0.1], 0.75, 0.05, 0.005,
                                       transparency_scorer=t_scorer, transparency_weight=0.1)[0]
        assert r_with_t.transparency_score is not None
        # With positive transparency score, total should be higher
        assert r_with_t.total_score != r_no_t.total_score

    def test_satisfies_protocol(self):
        from src.accelerate.batch_candidate_scorer import BatchCandidateScorer, PythonBatchCandidateScorer
        scorer = self._build_scorer()
        assert isinstance(scorer, BatchCandidateScorer)
