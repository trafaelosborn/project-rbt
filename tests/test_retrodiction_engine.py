"""Tests for src.retrodiction.engine (unit-level, no disk I/O to real corpora)"""

import json
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.retrodiction.engine import (
    RetrodictionConfig,
    BridgeStageRecord,
    RetrodictionEngine,
    LANG_CODES,
)


def _tiny_sequences(n=50, vocab=("a", "b", "c", "d"), length=5, seed=0):
    rng = np.random.default_rng(seed)
    return [list(rng.choice(list(vocab), size=length)) for _ in range(n)]


class TestRetrodictionConfig:
    def test_defaults(self):
        cfg = RetrodictionConfig()
        assert 0 < cfg.alpha < 1
        assert cfg.num_sequences > 0
        assert cfg.max_iterations > 0
        assert cfg.stability_threshold > 0

    def test_to_dict_has_all_keys(self):
        cfg = RetrodictionConfig()
        d = cfg.to_dict()
        assert set(d) == {"alpha", "num_sequences", "max_iterations", "stability_threshold", "seed"}


class TestBridgeStageRecord:
    def _make_record(self):
        return BridgeStageRecord(
            stage_id="FR_retro_000",
            source_language="french",
            iteration=0,
            alpha_cumulative=0.0,
            fingerprint_paths={
                "cooccurrence_matrix": "data/retrodiction/french/matrices/FR_retro_000_cooccurrence.npy",
                "positional_dist": "data/retrodiction/french/matrices/FR_retro_000_positional.npy",
                "ngram_meta": "data/retrodiction/french/matrices/FR_retro_000_ngram_meta.json",
            },
            type_token_ratio=0.13,
            bigram_coverage=0.35,
            trigram_coverage=0.25,
            bigram_entropy=4.5,
            trigram_entropy=5.8,
            scores={
                "vs_markov_noise": 0.7,
                "vs_sumerian": 0.6,
                "vs_portuguese_control": None,
                "vs_latin_ground_truth": None,
            },
            structural_vector=[0.13, 0.35, 0.25, 2.3],
        )

    def test_to_dict_has_required_keys(self):
        record = self._make_record()
        d = record.to_dict()
        assert "stage_id" in d
        assert "source_language" in d
        assert "iteration" in d
        assert "fingerprint" in d
        assert "scores" in d
        assert "structural_vector" in d

    def test_scores_has_all_four_fields(self):
        record = self._make_record()
        scores = record.to_dict()["scores"]
        assert "vs_markov_noise" in scores
        assert "vs_sumerian" in scores
        assert "vs_portuguese_control" in scores
        assert "vs_latin_ground_truth" in scores

    def test_portuguese_and_latin_null_initially(self):
        record = self._make_record()
        scores = record.to_dict()["scores"]
        assert scores["vs_portuguese_control"] is None
        assert scores["vs_latin_ground_truth"] is None

    def test_save_and_load(self, tmp_path):
        record = self._make_record()
        path = tmp_path / "FR_retro_000.json"
        record.save(path)
        with path.open() as fh:
            loaded = json.load(fh)
        assert loaded["stage_id"] == "FR_retro_000"
        assert loaded["scores"]["vs_portuguese_control"] is None


class TestLangCodes:
    def test_phase3_languages_have_codes(self):
        for lang in ["french", "italian", "spanish", "romanian"]:
            assert lang in LANG_CODES

    def test_codes_are_uppercase_strings(self):
        for code in LANG_CODES.values():
            assert code == code.upper()


class TestRetrodictionEngineSmoke:
    """Smoke test: run a very short retrodiction with tiny synthetic data."""

    def test_run_produces_records(self, tmp_path):
        seqs = _tiny_sequences(n=100)
        cfg = RetrodictionConfig(
            alpha=0.5,
            num_sequences=50,
            max_iterations=3,
            stability_threshold=0.0,   # never halt early
            seed=0,
        )
        engine = RetrodictionEngine(
            language="french",
            source_sequences=seqs,
            config=cfg,
            output_dir=tmp_path / "retrodiction" / "french",
        )
        records = engine.run()
        assert len(records) == 3
        assert all(isinstance(r, BridgeStageRecord) for r in records)

    def test_stage_ids_sequential(self, tmp_path):
        seqs = _tiny_sequences(n=100)
        cfg = RetrodictionConfig(alpha=0.5, num_sequences=50, max_iterations=3,
                                  stability_threshold=0.0, seed=0)
        engine = RetrodictionEngine("french", seqs, cfg, tmp_path / "out")
        records = engine.run()
        assert records[0].stage_id == "FR_retro_000"
        assert records[1].stage_id == "FR_retro_001"
        assert records[2].stage_id == "FR_retro_002"

    def test_records_saved_to_disk(self, tmp_path):
        seqs = _tiny_sequences(n=100)
        cfg = RetrodictionConfig(alpha=0.5, num_sequences=50, max_iterations=2,
                                  stability_threshold=0.0, seed=0)
        engine = RetrodictionEngine("french", seqs, cfg, tmp_path / "out")
        engine.run()
        record_files = list((tmp_path / "out" / "records").glob("*.json"))
        assert len(record_files) == 2

    def test_run_summary_saved(self, tmp_path):
        seqs = _tiny_sequences(n=100)
        cfg = RetrodictionConfig(alpha=0.5, num_sequences=50, max_iterations=2,
                                  stability_threshold=0.0, seed=0)
        engine = RetrodictionEngine("french", seqs, cfg, tmp_path / "out")
        engine.run()
        summary_path = tmp_path / "out" / "run_summary.json"
        assert summary_path.exists()
        with summary_path.open() as fh:
            summary = json.load(fh)
        assert summary["language"] == "french"
        assert summary["total_stages"] == 2

    def test_stability_halts_run(self, tmp_path):
        seqs = _tiny_sequences(n=100)
        cfg = RetrodictionConfig(
            alpha=0.0,
            num_sequences=50,
            max_iterations=50,
            # Very high threshold: any delta will be below it, halts at iteration 1
            stability_threshold=1000.0,
            seed=0,
        )
        engine = RetrodictionEngine("french", seqs, cfg, tmp_path / "out")
        records = engine.run()
        # Should halt at iteration 1 (first time prev_vec is available)
        assert len(records) == 2
        assert "stable" in records[-1].flags

    def test_bigram_entropy_increases_with_mixing(self, tmp_path):
        seqs = _tiny_sequences(n=200, vocab=("a","b","c","d","e"), length=8)
        cfg = RetrodictionConfig(alpha=0.3, num_sequences=200, max_iterations=5,
                                  stability_threshold=0.0, seed=0)
        engine = RetrodictionEngine("french", seqs, cfg, tmp_path / "out")
        records = engine.run()
        entropies = [r.bigram_entropy for r in records]
        # Entropy should generally increase (or at least not dramatically decrease)
        assert entropies[-1] >= entropies[0] - 0.5  # allow some sampling noise
