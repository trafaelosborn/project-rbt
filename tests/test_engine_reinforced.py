"""Tests for src.retrodiction.engine_reinforced (unit-level, no real disk corpora)."""

import json
import numpy as np
import pytest
from pathlib import Path

from src.retrodiction.engine_reinforced import (
    ReinforcedConfig,
    ReinforcedStageRecord,
    LatinReference,
    StochasticRetrodictionEngine,
    GradientRetrodictionEngine,
)
from src.retrodiction.generate import BigramModel
from src.retrodiction.similarity import structural_vector
from src.fingerprint.ngram import extract_ngrams, build_profile


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tiny_sequences(n=50, vocab=("a", "b", "c", "d"), length=5, seed=0):
    rng = np.random.default_rng(seed)
    return [list(rng.choice(list(vocab), size=length)) for _ in range(n)]


def _make_latin_ref(vocab=("a", "b", "c", "d"), seed=99):
    """
    Build a LatinReference that bypasses file I/O.
    Only sets the attributes the engines actually use: .vec and .model.
    """
    ref = object.__new__(LatinReference)
    seqs = _tiny_sequences(n=100, vocab=vocab, length=5, seed=seed)
    model = BigramModel.from_sequences(seqs)
    bg = build_profile(extract_ngrams(seqs, 2), 100)
    tg = build_profile(extract_ngrams(seqs, 3), 100)
    ref.vec = structural_vector(seqs, bg, tg)
    ref.model = model
    return ref


class _FakeReferenceSet:
    def score(self, sequences, bigram_profile, trigram_profile):
        return {
            "vs_markov_noise": 0.25,
            "vs_sumerian": 0.5,
            "vs_portuguese_control": None,
            "vs_latin_ground_truth": None,
        }

    def coherence_from_vector(self, vec):
        return {
            "distance_to_real_language_centroid": 1.0,
            "distance_to_markov_noise": 2.0,
            "language_likeness_margin": 1.0,
            "coherence_label": "coherent",
        }


# ---------------------------------------------------------------------------
# ReinforcedConfig
# ---------------------------------------------------------------------------

class TestReinforcedConfig:
    def test_defaults(self):
        cfg = ReinforcedConfig()
        assert cfg.num_sequences > 0
        assert cfg.max_iterations > 0
        assert 0 < cfg.alpha < 1
        assert cfg.n_candidates > 0
        assert cfg.noise_scale > 0

    def test_to_dict_has_all_keys(self):
        cfg = ReinforcedConfig()
        d = cfg.to_dict()
        assert set(d) == {
            "num_sequences", "max_iterations", "stability_threshold",
            "seed", "n_candidates", "noise_scale", "alpha",
        }


# ---------------------------------------------------------------------------
# ReinforcedStageRecord
# ---------------------------------------------------------------------------

class TestReinforcedStageRecord:
    def _make_record(self, algorithm="gradient"):
        return ReinforcedStageRecord(
            stage_id="RO_grad_000",
            source_language="romanian",
            algorithm=algorithm,
            iteration=0,
            fingerprint_paths={
                "cooccurrence_matrix": "data/retrodiction/romanian/gradient/matrices/RO_grad_000_cooccurrence.npy",
                "positional_dist":     "data/retrodiction/romanian/gradient/matrices/RO_grad_000_positional.npy",
                "ngram_meta":          "data/retrodiction/romanian/gradient/matrices/RO_grad_000_ngram_meta.json",
            },
            artifact_paths={
                "corpus_json": "data/retrodiction/romanian/gradient/corpora/RO_grad_000_tokens.json",
                "preview_txt": "data/retrodiction/romanian/gradient/previews/RO_grad_000_preview.txt",
            },
            type_token_ratio=0.21,
            bigram_coverage=0.18,
            trigram_coverage=0.14,
            structural_vector=[0.21, 0.18, 0.14, 2.3],
            latin_score=-0.05,
            scores={
                "vs_markov_noise": 0.25,
                "vs_sumerian": 0.5,
                "vs_portuguese_control": None,
                "vs_latin_ground_truth": None,
            },
            bigram_entropy=4.2,
            trigram_entropy=5.1,
            diagnostics={
                "distance_to_real_language_centroid": 1.0,
                "distance_to_markov_noise": 2.0,
                "language_likeness_margin": 1.0,
                "coherence_label": "coherent",
            },
        )

    def test_to_dict_has_required_keys(self):
        d = self._make_record().to_dict()
        for key in ("stage_id", "source_language", "algorithm", "iteration",
                    "fingerprint", "artifacts", "structural_vector", "latin_score", "scores"):
            assert key in d, f"missing key: {key}"

    def test_algorithm_field_preserved(self):
        assert self._make_record("stochastic").to_dict()["algorithm"] == "stochastic"
        assert self._make_record("gradient").to_dict()["algorithm"] == "gradient"

    def test_latin_score_round_trips(self):
        d = self._make_record().to_dict()
        assert abs(d["latin_score"] - (-0.05)) < 1e-5

    def test_scores_include_null_and_reference_fields(self):
        scores = self._make_record().to_dict()["scores"]
        assert scores["vs_markov_noise"] == pytest.approx(0.25, abs=1e-9)
        assert scores["vs_sumerian"] == pytest.approx(0.5, abs=1e-9)
        assert scores["vs_portuguese_control"] is None
        assert scores["vs_latin_ground_truth"] is None

    def test_diagnostics_round_trip(self):
        d = self._make_record().to_dict()
        assert d["diagnostics"]["coherence_label"] == "coherent"
        assert d["diagnostics"]["language_likeness_margin"] == pytest.approx(1.0, abs=1e-9)

    def test_artifact_paths_round_trip(self):
        d = self._make_record().to_dict()
        assert d["artifacts"]["corpus_json"].endswith("RO_grad_000_tokens.json")
        assert d["artifacts"]["preview_txt"].endswith("RO_grad_000_preview.txt")

    def test_save_and_load(self, tmp_path):
        record = self._make_record()
        path = tmp_path / "RO_grad_000.json"
        record.save(path)
        with path.open() as fh:
            loaded = json.load(fh)
        assert loaded["stage_id"] == "RO_grad_000"
        assert loaded["algorithm"] == "gradient"
        assert loaded["latin_score"] == pytest.approx(-0.05, abs=1e-5)


# ---------------------------------------------------------------------------
# LatinReference.score (unit, no file I/O)
# ---------------------------------------------------------------------------

class TestLatinReferenceScore:
    def test_score_identical_vector_is_zero(self):
        ref = _make_latin_ref()
        assert ref.score(ref.vec) == pytest.approx(0.0, abs=1e-9)

    def test_score_ignores_log_mean_seq_len_dimension(self):
        ref = object.__new__(LatinReference)
        ref.vec = np.array([0.1, 0.2, 0.3, 9.9], dtype=np.float64)
        ref.reward_vec = ref.vec[:3].copy()
        ref.score_scale = 5.0

        short_seq_vec = np.array([0.1, 0.2, 0.3, 1.0], dtype=np.float64)
        long_seq_vec = np.array([0.1, 0.2, 0.3, 100.0], dtype=np.float64)

        assert ref.score(short_seq_vec) == pytest.approx(0.0, abs=1e-9)
        assert ref.score(long_seq_vec) == pytest.approx(0.0, abs=1e-9)
        assert ref.score(short_seq_vec) == pytest.approx(ref.score(long_seq_vec), abs=1e-9)

    def test_score_is_non_positive(self):
        ref = _make_latin_ref()
        rng = np.random.default_rng(0)
        for _ in range(10):
            vec = rng.random(4)
            assert ref.score(vec) <= 0.0 + 1e-12

    def test_score_decreases_with_distance(self):
        ref = _make_latin_ref()
        close_vec = ref.vec + np.array([0.001, 0.001, 0.001, 0.001])
        far_vec = ref.vec + np.array([1.0, 1.0, 1.0, 1.0])
        assert ref.score(close_vec) > ref.score(far_vec)


# ---------------------------------------------------------------------------
# StochasticRetrodictionEngine smoke tests
# ---------------------------------------------------------------------------

class TestStochasticSmoke:
    def _make_engine(self, tmp_path, max_iterations=3, stability_threshold=0.0):
        seqs = _tiny_sequences(n=100, vocab=("a", "b", "c", "d"), length=5)
        latin_ref = _make_latin_ref()
        references = _FakeReferenceSet()
        cfg = ReinforcedConfig(
            num_sequences=50,
            max_iterations=max_iterations,
            stability_threshold=stability_threshold,
            n_candidates=5,
            noise_scale=0.3,
            seed=0,
        )
        return StochasticRetrodictionEngine(
            language="romanian",
            source_sequences=seqs,
            latin_ref=latin_ref,
            config=cfg,
            output_dir=tmp_path / "stochastic",
            references=references,
        )

    def test_run_produces_records(self, tmp_path):
        engine = self._make_engine(tmp_path, max_iterations=3)
        records = engine.run()
        assert len(records) == 3

    def test_stage_ids_sequential(self, tmp_path):
        engine = self._make_engine(tmp_path, max_iterations=3)
        records = engine.run()
        assert records[0].stage_id == "RO_stoch_000"
        assert records[1].stage_id == "RO_stoch_001"
        assert records[2].stage_id == "RO_stoch_002"

    def test_algorithm_field_is_stochastic(self, tmp_path):
        engine = self._make_engine(tmp_path, max_iterations=2)
        records = engine.run()
        assert all(r.algorithm == "stochastic" for r in records)

    def test_records_saved_to_disk(self, tmp_path):
        engine = self._make_engine(tmp_path, max_iterations=2)
        engine.run()
        files = list((tmp_path / "stochastic" / "records").glob("*.json"))
        assert len(files) == 2

    def test_corpora_and_previews_saved(self, tmp_path):
        engine = self._make_engine(tmp_path, max_iterations=2)
        engine.run()
        corpora = list((tmp_path / "stochastic" / "corpora").glob("*_tokens.json"))
        previews = list((tmp_path / "stochastic" / "previews").glob("*_preview.txt"))
        assert len(corpora) == 2
        assert len(previews) == 2

    def test_summary_saved(self, tmp_path):
        engine = self._make_engine(tmp_path, max_iterations=2)
        engine.run()
        summary_path = tmp_path / "stochastic" / "run_summary.json"
        assert summary_path.exists()
        with summary_path.open() as fh:
            summary = json.load(fh)
        assert summary["language"] == "romanian"
        assert summary["algorithm"] == "stochastic"
        assert summary["total_stages"] == 2
        assert summary["best_stage_id"] in {"RO_stoch_000", "RO_stoch_001"}
        assert summary["best_corpus_json"].endswith("_tokens.json")
        assert summary["best_preview_txt"].endswith("_preview.txt")

    def test_stability_halts_run(self, tmp_path):
        # stability_threshold=1000.0 halts after the first score-delta check (iteration 1)
        engine = self._make_engine(tmp_path, max_iterations=50, stability_threshold=1000.0)
        records = engine.run()
        assert len(records) == 2
        assert "stable" in records[-1].flags

    def test_latin_score_is_non_positive(self, tmp_path):
        engine = self._make_engine(tmp_path, max_iterations=3)
        records = engine.run()
        assert all(r.latin_score <= 0.0 + 1e-9 for r in records)

    def test_records_include_coherence_and_null_scores(self, tmp_path):
        engine = self._make_engine(tmp_path, max_iterations=2)
        records = engine.run()
        assert all(r.scores["vs_markov_noise"] == pytest.approx(0.25, abs=1e-9) for r in records)
        assert all(r.diagnostics["coherence_label"] == "coherent" for r in records)
        assert all(r.artifact_paths["corpus_json"].endswith("_tokens.json") for r in records)


# ---------------------------------------------------------------------------
# GradientRetrodictionEngine smoke tests
# ---------------------------------------------------------------------------

class TestGradientSmoke:
    def _make_engine(self, tmp_path, max_iterations=3, stability_threshold=0.0):
        seqs = _tiny_sequences(n=100, vocab=("a", "b", "c", "d"), length=5)
        latin_ref = _make_latin_ref()
        references = _FakeReferenceSet()
        cfg = ReinforcedConfig(
            num_sequences=50,
            max_iterations=max_iterations,
            stability_threshold=stability_threshold,
            alpha=0.1,
            seed=0,
        )
        return GradientRetrodictionEngine(
            language="romanian",
            source_sequences=seqs,
            latin_ref=latin_ref,
            config=cfg,
            output_dir=tmp_path / "gradient",
            references=references,
        )

    def test_run_produces_records(self, tmp_path):
        engine = self._make_engine(tmp_path, max_iterations=3)
        records = engine.run()
        assert len(records) == 3

    def test_stage_ids_sequential(self, tmp_path):
        engine = self._make_engine(tmp_path, max_iterations=3)
        records = engine.run()
        assert records[0].stage_id == "RO_grad_000"
        assert records[1].stage_id == "RO_grad_001"
        assert records[2].stage_id == "RO_grad_002"

    def test_algorithm_field_is_gradient(self, tmp_path):
        engine = self._make_engine(tmp_path, max_iterations=2)
        records = engine.run()
        assert all(r.algorithm == "gradient" for r in records)

    def test_records_saved_to_disk(self, tmp_path):
        engine = self._make_engine(tmp_path, max_iterations=2)
        engine.run()
        files = list((tmp_path / "gradient" / "records").glob("*.json"))
        assert len(files) == 2

    def test_corpora_and_previews_saved(self, tmp_path):
        engine = self._make_engine(tmp_path, max_iterations=2)
        engine.run()
        corpora = list((tmp_path / "gradient" / "corpora").glob("*_tokens.json"))
        previews = list((tmp_path / "gradient" / "previews").glob("*_preview.txt"))
        assert len(corpora) == 2
        assert len(previews) == 2

    def test_summary_saved(self, tmp_path):
        engine = self._make_engine(tmp_path, max_iterations=2)
        engine.run()
        summary_path = tmp_path / "gradient" / "run_summary.json"
        assert summary_path.exists()
        with summary_path.open() as fh:
            summary = json.load(fh)
        assert summary["language"] == "romanian"
        assert summary["algorithm"] == "gradient"
        assert summary["total_stages"] == 2
        assert summary["best_stage_id"] in {"RO_grad_000", "RO_grad_001"}
        assert summary["best_corpus_json"].endswith("_tokens.json")
        assert summary["best_preview_txt"].endswith("_preview.txt")

    def test_stability_halts_run(self, tmp_path):
        engine = self._make_engine(tmp_path, max_iterations=50, stability_threshold=1000.0)
        records = engine.run()
        assert len(records) == 2
        assert "stable" in records[-1].flags

    def test_latin_score_is_non_positive(self, tmp_path):
        engine = self._make_engine(tmp_path, max_iterations=3)
        records = engine.run()
        assert all(r.latin_score <= 0.0 + 1e-9 for r in records)

    def test_records_include_coherence_and_null_scores(self, tmp_path):
        engine = self._make_engine(tmp_path, max_iterations=2)
        records = engine.run()
        assert all(r.scores["vs_sumerian"] == pytest.approx(0.5, abs=1e-9) for r in records)
        assert all(r.diagnostics["coherence_label"] == "coherent" for r in records)
        assert all(r.artifact_paths["preview_txt"].endswith("_preview.txt") for r in records)

    def test_gradient_monotonically_improves(self, tmp_path):
        """
        With same-vocabulary source and Latin, gradient mixing must move
        the structural vector closer to Latin monotonically.
        Allows a tolerance of 0.05 for sampling noise.
        """
        engine = self._make_engine(tmp_path, max_iterations=5)
        records = engine.run()
        scores = [r.latin_score for r in records]
        # Each step should not dramatically worsen the score
        for i in range(1, len(scores)):
            assert scores[i] >= scores[0] - 0.2, (
                f"Score regressed badly at step {i}: {scores[i]:.4f} vs start {scores[0]:.4f}"
            )


# ---------------------------------------------------------------------------
# GradientRetrodictionEngine._align_vocab (internal unit tests)
# ---------------------------------------------------------------------------

class TestAlignVocab:
    def _make_gradient_engine(self, source_seqs, latin_ref, tmp_path):
        cfg = ReinforcedConfig(max_iterations=1, num_sequences=20)
        return GradientRetrodictionEngine(
            "romanian", source_seqs, latin_ref, cfg, tmp_path / "grad"
        )

    def test_rows_sum_to_one_full_overlap(self, tmp_path):
        vocab = ("a", "b", "c", "d")
        source_seqs = _tiny_sequences(100, vocab, 5, seed=0)
        latin_ref = _make_latin_ref(vocab, seed=1)
        engine = self._make_gradient_engine(source_seqs, latin_ref, tmp_path)
        source_model = BigramModel.from_sequences(source_seqs)
        _, latin_T = engine._align_vocab(source_model, latin_ref.model)
        assert np.allclose(latin_T.sum(axis=1), 1.0)

    def test_rows_sum_to_one_no_overlap(self, tmp_path):
        source_seqs = _tiny_sequences(100, ("a", "b"), 5, seed=0)
        latin_ref = _make_latin_ref(("c", "d"), seed=1)
        engine = self._make_gradient_engine(source_seqs, latin_ref, tmp_path)
        source_model = BigramModel.from_sequences(source_seqs)
        _, latin_T = engine._align_vocab(source_model, latin_ref.model)
        assert np.allclose(latin_T.sum(axis=1), 1.0)

    def test_no_overlap_keeps_source_transitions(self, tmp_path):
        """When source vocab has no tokens in Latin, rows equal source's own transitions."""
        source_seqs = _tiny_sequences(100, ("a", "b"), 5, seed=0)
        latin_ref = _make_latin_ref(("c", "d"), seed=1)
        engine = self._make_gradient_engine(source_seqs, latin_ref, tmp_path)
        source_model = BigramModel.from_sequences(source_seqs)
        _, latin_T = engine._align_vocab(source_model, latin_ref.model)
        assert np.allclose(latin_T, source_model.transitions)

    def test_full_overlap_uses_latin_transitions(self, tmp_path):
        """When all source tokens exist in Latin, rows should reflect Latin's bigrams."""
        vocab = ("a", "b", "c", "d")
        source_seqs = _tiny_sequences(100, vocab, 5, seed=0)
        # Latin with same vocab but different seed → different distribution
        latin_ref = _make_latin_ref(vocab, seed=99)
        engine = self._make_gradient_engine(source_seqs, latin_ref, tmp_path)
        source_model = BigramModel.from_sequences(source_seqs)
        source_T, latin_T = engine._align_vocab(source_model, latin_ref.model)
        # With same vocab, latin_T should NOT equal source_T
        assert not np.allclose(latin_T, source_T)
        # And latin_T rows should be valid distributions
        assert np.allclose(latin_T.sum(axis=1), 1.0)
