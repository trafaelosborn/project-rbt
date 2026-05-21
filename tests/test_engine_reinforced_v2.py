"""Tests for src.retrodiction.engine_reinforced_v2."""

import json
import shutil
from pathlib import Path

import numpy as np

from src.retrodiction.engine_reinforced_v2 import (
    ReinforcedV2Config,
    RelationalReinforcedRetrodictionEngine,
)


def _tiny_sequences():
    return [
        ["des", "les", "par", "plus", "nation"],
        ["des", "les", "par", "chantes", "relation"],
        ["des", "les", "par", "plus", "union"],
        ["des", "les", "par", "mangent", "question"],
        ["des", "les", "par", "plus", "vision"],
    ] * 24


class _FakeStructuralRef:
    def score(self, vec):
        return float(-0.5 * abs(vec[0] - 0.3) - 0.5 * abs(vec[1] - 0.2) - 0.5 * abs(vec[2] - 0.1))


class _FakeLatinFormRef:
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


class TestReinforcedV2Config:
    def test_to_dict_has_new_search_keys(self):
        cfg = ReinforcedV2Config()
        data = cfg.to_dict()
        assert data["token_edit_attempts"] == cfg.token_edit_attempts
        assert data["suffix_candidate_samples"] == cfg.suffix_candidate_samples
        assert data["span_min_sequences"] == cfg.span_min_sequences
        assert data["span_max_sequences"] == cfg.span_max_sequences
        assert data["span_edit_min"] == cfg.span_edit_min
        assert data["span_edit_max"] == cfg.span_edit_max
        assert data["mutation_cost_weight"] == cfg.mutation_cost_weight


class TestRelationalReinforcedV2Smoke:
    def test_run_accepts_mutations_and_writes_artifacts(self):
        output_dir = (Path.cwd() / "data" / "retrodiction" / "_test_v2_smoke").resolve()
        try:
            shutil.rmtree(output_dir, ignore_errors=True)
            engine = RelationalReinforcedRetrodictionEngine(
                language="french",
                source_sequences=_tiny_sequences(),
                latin_structural_ref=_FakeStructuralRef(),
                latin_form_ref=_FakeLatinFormRef(),
                config=ReinforcedV2Config(
                    num_sequences=60,
                    max_proposals=12,
                    max_accepted_stages=6,
                    patience=4,
                    seed=0,
                    n_candidates=4,
                    min_improvement=0.0001,
                ),
                output_dir=output_dir,
                references=_FakeReferenceSet(),
            )

            records = engine.run()

            assert len(records) > 1
            assert any(record.mutation_operator != "seed" for record in records[1:])

            summary_path = output_dir / "run_summary.json"
            assert summary_path.exists()

            with summary_path.open(encoding="utf-8") as fh:
                summary = json.load(fh)

            assert summary["accepted_mutation_stages"] >= 1
            assert Path(summary["best_corpus_json"]).exists()
            assert Path(summary["best_preview_txt"]).exists()
            assert summary["final_coherence_label"] == "coherent"
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)


class TestRelationalReinforcedV2SpanMutation:
    def test_sequence_span_rewrite_mutates_multiple_sequences(self):
        output_dir = (Path.cwd() / "data" / "retrodiction" / "_test_v2_span").resolve()
        try:
            shutil.rmtree(output_dir, ignore_errors=True)
            engine = RelationalReinforcedRetrodictionEngine(
                language="french",
                source_sequences=_tiny_sequences(),
                latin_structural_ref=_FakeStructuralRef(),
                latin_form_ref=_FakeLatinFormRef(),
                config=ReinforcedV2Config(
                    num_sequences=60,
                    span_min_sequences=2,
                    span_max_sequences=4,
                    span_edit_min=2,
                    span_edit_max=3,
                ),
                output_dir=output_dir,
                references=_FakeReferenceSet(),
            )
            rng = np.random.default_rng(7)
            mutated, details, cost = engine._mutate_sequence_span_rewrite(_tiny_sequences(), rng)

            assert mutated is not None
            assert mutated != _tiny_sequences()
            assert details.startswith("span[")
            assert cost > 0.75
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)

    def test_run_can_accept_span_rewrite_operator(self):
        output_dir = (Path.cwd() / "data" / "retrodiction" / "_test_v2_span_run").resolve()
        try:
            shutil.rmtree(output_dir, ignore_errors=True)
            engine = RelationalReinforcedRetrodictionEngine(
                language="french",
                source_sequences=_tiny_sequences(),
                latin_structural_ref=_FakeStructuralRef(),
                latin_form_ref=_FakeLatinFormRef(),
                config=ReinforcedV2Config(
                    num_sequences=60,
                    max_proposals=20,
                    max_accepted_stages=6,
                    patience=6,
                    seed=3,
                    n_candidates=4,
                    min_improvement=0.0001,
                    operator_weights=(0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
                ),
                output_dir=output_dir,
                references=_FakeReferenceSet(),
            )

            records = engine.run()
            assert len(records) > 1
            assert any(record.mutation_operator == "sequence_span_rewrite" for record in records[1:])

            with (output_dir / "run_summary.json").open(encoding="utf-8") as fh:
                summary = json.load(fh)
            assert summary["accepted_operator_counts"].get("sequence_span_rewrite", 0) >= 1
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)
