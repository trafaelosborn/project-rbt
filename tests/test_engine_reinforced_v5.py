"""Tests for src.retrodiction.engine_reinforced_v5."""

import json
import shutil
from pathlib import Path

import numpy as np

from src.retrodiction.engine_reinforced_v5 import (
    ReinforcedV5Config,
    RelationalReinforcedRetrodictionEngineV5,
)
from src.validation.hungarian_alignment import extract_family_inventory


def _tiny_sequences():
    return [
        ["des", "les", "par", "plus", "nation"],
        ["des", "les", "par", "chantes", "relation"],
        ["des", "les", "par", "plus", "union"],
        ["des", "les", "par", "mangent", "question"],
        ["des", "les", "par", "plus", "vision"],
    ] * 24


def _latinish_sequences():
    return [
        ["de", "la", "forum", "bellum", "datum", "romanum"],
        ["de", "la", "consilium", "conventum", "ludum", "datum"],
        ["de", "la", "bellum", "forum", "romanum", "consilium"],
    ] * 24


class _FakeStructuralRef:
    def score(self, vec):
        return float(-0.5 * abs(vec[0] - 0.3) - 0.5 * abs(vec[1] - 0.2) - 0.5 * abs(vec[2] - 0.1))


class _FakeLatinFormRef:
    char_bigram_profile = {}
    char_trigram_profile = {}
    suffix_profile = {}

    def sample_char(self, rng):
        return "u"

    def sample_suffix(self, rng):
        return "um"

    def score_token(self, token):
        score = 0.0
        if "u" in token:
            score += 0.2
        if token.endswith("um"):
            score += 0.8
        if token.endswith("us"):
            score += 0.6
        if token.endswith("ere"):
            score += 0.4
        if token.endswith("tus"):
            score += 0.5
        return score

    def score(self, sequences):
        tokens = [tok for seq in sequences for tok in seq]
        total = sum(self.score_token(tok) for tok in tokens) / max(len(tokens), 1)
        return {
            "latin_form_score": float(total),
            "latin_char_bigram_cosine": float(total),
            "latin_char_trigram_cosine": float(total),
            "latin_suffix_cosine": float(total),
        }


class _FakeReferenceSet:
    markov = np.array([1.0, 1.0, 1.0, 1.0])
    sumerian = np.array([0.5, 0.5, 0.5, 0.5])

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
            "distance_to_markov_noise": 3.0,
            "language_likeness_margin": 2.0,
            "coherence_label": "coherent",
        }


class TestReinforcedV5Config:
    def test_to_dict_includes_shock_keys(self):
        cfg = ReinforcedV5Config()
        data = cfg.to_dict()
        assert data["shock_plateau_window"] == cfg.shock_plateau_window
        assert data["max_culture_bombs"] == cfg.max_culture_bombs
        assert data["culture_bomb_cost_discount"] == cfg.culture_bomb_cost_discount


class TestRelationalReinforcedV5:
    def test_culture_bomb_can_mutate_sequences(self):
        output_dir = (Path.cwd() / "data" / "retrodiction" / "_test_v5_bomb").resolve()
        try:
            shutil.rmtree(output_dir, ignore_errors=True)
            cfg = ReinforcedV5Config(seed=3)
            family_ref = extract_family_inventory("latin", _latinish_sequences(), cfg.alignment_config)
            engine = RelationalReinforcedRetrodictionEngineV5(
                language="french",
                source_sequences=_tiny_sequences(),
                latin_structural_ref=_FakeStructuralRef(),
                latin_form_ref=_FakeLatinFormRef(),
                config=cfg,
                output_dir=output_dir,
                references=_FakeReferenceSet(),
                family_reference_inventory=family_ref,
            )
            mutated, details, cost = engine._mutate_culture_bomb(_tiny_sequences(), np.random.default_rng(3))
            assert mutated is not None
            assert mutated != _tiny_sequences()
            assert details
            assert cost > 0.0
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)

    def test_run_writes_v5_summary(self):
        output_dir = (Path.cwd() / "data" / "retrodiction" / "_test_v5_smoke").resolve()
        try:
            shutil.rmtree(output_dir, ignore_errors=True)
            cfg = ReinforcedV5Config(
                num_sequences=60,
                max_proposals=10,
                max_accepted_stages=6,
                patience=4,
                enable_culture_bombs=True,
                shock_plateau_window=3,
                max_culture_bombs=2,
                culture_bomb_candidates=3,
                seed=2,
                n_candidates=4,
                min_improvement=0.00005,
            )
            family_ref = extract_family_inventory("latin", _latinish_sequences(), cfg.alignment_config)
            engine = RelationalReinforcedRetrodictionEngineV5(
                language="french",
                source_sequences=_tiny_sequences(),
                latin_structural_ref=_FakeStructuralRef(),
                latin_form_ref=_FakeLatinFormRef(),
                config=cfg,
                output_dir=output_dir,
                references=_FakeReferenceSet(),
                family_reference_inventory=family_ref,
            )

            records = engine.run()
            assert len(records) >= 1
            with (output_dir / "run_summary.json").open(encoding="utf-8") as fh:
                summary = json.load(fh)
            assert summary["algorithm"] == "relational_v5"
            assert "culture_bombs_used" in summary
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)

    def test_plain_v5_writes_live_event_stream(self):
        output_dir = (Path.cwd() / "data" / "retrodiction" / "_test_v5_plain_events").resolve()
        try:
            shutil.rmtree(output_dir, ignore_errors=True)
            cfg = ReinforcedV5Config(
                num_sequences=60,
                max_proposals=6,
                max_accepted_stages=4,
                patience=4,
                seed=5,
                n_candidates=3,
                min_improvement=0.00005,
                enable_culture_bombs=False,
                use_incremental_scoring=True,
            )
            family_ref = extract_family_inventory("latin", _latinish_sequences(), cfg.alignment_config)
            engine = RelationalReinforcedRetrodictionEngineV5(
                language="french",
                source_sequences=_tiny_sequences(),
                latin_structural_ref=_FakeStructuralRef(),
                latin_form_ref=_FakeLatinFormRef(),
                config=cfg,
                output_dir=output_dir,
                references=_FakeReferenceSet(),
                family_reference_inventory=family_ref,
            )

            engine.run()
            events_path = output_dir / "live_events.jsonl"
            assert events_path.exists()
            lines = [line for line in events_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            assert lines
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)
