"""Tests for src.retrodiction.engine_reinforced_v3."""

import json
import shutil
from pathlib import Path

import numpy as np

from src.retrodiction.engine_reinforced_v2 import CandidateState
from src.retrodiction.engine_reinforced_v3 import (
    ReinforcedV3Config,
    RelationalReinforcedRetrodictionEngineV3,
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


class TestReinforcedV3Config:
    def test_to_dict_includes_v3_reward_and_mutation_keys(self):
        cfg = ReinforcedV3Config()
        data = cfg.to_dict()
        assert data["function_burst_min_tokens"] == cfg.function_burst_min_tokens
        assert data["macro_bundle_min_steps"] == cfg.macro_bundle_min_steps
        assert data["reward_struct_gain_weight"] == cfg.reward_struct_gain_weight
        assert data["reward_joint_bonus"] == cfg.reward_joint_bonus
        assert len(data["operator_weights"]) == 9


class TestRelationalReinforcedV3:
    def test_function_word_burst_mutates_short_tokens(self):
        output_dir = (Path.cwd() / "data" / "retrodiction" / "_test_v3_burst").resolve()
        try:
            shutil.rmtree(output_dir, ignore_errors=True)
            engine = RelationalReinforcedRetrodictionEngineV3(
                language="french",
                source_sequences=_tiny_sequences(),
                latin_structural_ref=_FakeStructuralRef(),
                latin_form_ref=_FakeLatinFormRef(),
                config=ReinforcedV3Config(
                    num_sequences=60,
                    function_burst_min_tokens=2,
                    function_burst_max_tokens=4,
                ),
                output_dir=output_dir,
                references=_FakeReferenceSet(),
            )
            rng = np.random.default_rng(5)
            mutated, details, cost = engine._mutate_function_word_burst(
                _tiny_sequences(),
                engine._token_counts(_tiny_sequences()),
                rng,
            )

            assert mutated is not None
            assert mutated != _tiny_sequences()
            assert "short tokens" in details
            assert cost > 0.5
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)

    def test_reward_amplification_boosts_jointly_good_candidate(self):
        output_dir = (Path.cwd() / "data" / "retrodiction" / "_test_v3_reward").resolve()
        try:
            shutil.rmtree(output_dir, ignore_errors=True)
            engine = RelationalReinforcedRetrodictionEngineV3(
                language="french",
                source_sequences=_tiny_sequences(),
                latin_structural_ref=_FakeStructuralRef(),
                latin_form_ref=_FakeLatinFormRef(),
                config=ReinforcedV3Config(),
                output_dir=output_dir,
                references=_FakeReferenceSet(),
            )

            current = CandidateState(
                sequences=[],
                operator="seed",
                details="",
                mutation_cost=0.0,
                structural_vector=np.zeros(4),
                latin_structural_score=-1.0,
                latin_form_score=0.50,
                form_details={
                    "latin_form_score": 0.50,
                    "latin_char_bigram_cosine": 0.40,
                    "latin_char_trigram_cosine": 0.30,
                    "latin_suffix_cosine": 0.20,
                },
                total_score=-0.7,
                scores={},
                diagnostics={"language_likeness_margin": 2.0, "coherence_label": "coherent"},
                type_token_ratio=0.0,
                bigram_coverage=0.0,
                trigram_coverage=0.0,
                bigram_profile={},
                trigram_profile={},
            )
            candidate = CandidateState(
                sequences=[],
                operator="candidate",
                details="",
                mutation_cost=1.0,
                structural_vector=np.zeros(4),
                latin_structural_score=-0.98,
                latin_form_score=0.55,
                form_details={
                    "latin_form_score": 0.55,
                    "latin_char_bigram_cosine": 0.43,
                    "latin_char_trigram_cosine": 0.34,
                    "latin_suffix_cosine": 0.28,
                },
                total_score=-0.69,
                scores={},
                diagnostics={"language_likeness_margin": 2.01, "coherence_label": "coherent"},
                type_token_ratio=0.0,
                bigram_coverage=0.0,
                trigram_coverage=0.0,
                bigram_profile={},
                trigram_profile={},
            )

            boosted = engine._amplify_reward(current, candidate)
            assert boosted.total_score > -0.60
            assert boosted.diagnostics["reward_bonus"] > 0.0
            assert boosted.diagnostics["reward_penalty_relief"] > 0.0
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)

    def test_run_accepts_function_word_burst_when_forced(self):
        output_dir = (Path.cwd() / "data" / "retrodiction" / "_test_v3_smoke").resolve()
        try:
            shutil.rmtree(output_dir, ignore_errors=True)
            engine = RelationalReinforcedRetrodictionEngineV3(
                language="french",
                source_sequences=_tiny_sequences(),
                latin_structural_ref=_FakeStructuralRef(),
                latin_form_ref=_FakeLatinFormRef(),
                config=ReinforcedV3Config(
                    num_sequences=60,
                    max_proposals=16,
                    max_accepted_stages=6,
                    patience=6,
                    seed=2,
                    n_candidates=4,
                    min_improvement=0.00005,
                    operator_weights=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0),
                ),
                output_dir=output_dir,
                references=_FakeReferenceSet(),
            )

            records = engine.run()
            assert len(records) > 1
            assert any(record.mutation_operator == "function_word_burst" for record in records[1:])

            with (output_dir / "run_summary.json").open(encoding="utf-8") as fh:
                summary = json.load(fh)
            assert summary["algorithm"] == "relational_v3"
            assert summary["accepted_operator_counts"].get("function_word_burst", 0) >= 1
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)
