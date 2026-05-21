"""Tests for src.retrodiction.engine_reinforced_v4."""

import json
import shutil
from pathlib import Path

import numpy as np

from src.accelerate.incremental_tensor_state import IncrementalFingerprintTensorState, TensorStateConfig
from src.accelerate.v4_batch_guidance import BatchGuidance, GuidanceAdjustment
from src.retrodiction.engine_reinforced_v2 import CandidateState
from src.retrodiction.engine_reinforced_v4 import (
    MutationPayload,
    ReinforcedV4Config,
    RelationalReinforcedRetrodictionEngineV4,
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


class _FakeGuidanceBuilder:
    def __init__(self):
        self.build_calls = 0
        self.build_from_state_calls = 0
        self.initial_state_builds = 0

    def _guidance(self, *, anchor_vocab_size=5):
        return BatchGuidance(
            backend_used="numpy",
            anchor_vocab_size=anchor_vocab_size,
            batch_size=6,
            selected_adjustments=(
                GuidanceAdjustment(
                    rank=1,
                    component_id=1,
                    component_name="cooccurrence",
                    row_index=0,
                    col_index=1,
                    row_token="des",
                    col_token="les",
                    signed_delta=1.25,
                    abs_score=1.25,
                ),
            ),
            hotspot_token_weights=(("des", 1.25), ("les", 0.75)),
            hotspot_pairs=(("des", "les", 1.25, 1.25),),
            positional_targets={"des": {0: 0.5}},
        )

    def build_initial_state(self, sequences):
        self.initial_state_builds += 1
        return IncrementalFingerprintTensorState.from_sequences(
            sequences,
            config=TensorStateConfig(max_vocab=128, cooccurrence_window=2),
            ngram_basis=None,
        )

    def build(self, sequences):
        self.build_calls += 1
        return self._guidance(anchor_vocab_size=5)

    def build_from_state(self, state):
        self.build_from_state_calls += 1
        return self._guidance(anchor_vocab_size=len(state.idx2token))


class TestReinforcedV4Config:
    def test_to_dict_includes_alignment_schedule_keys(self):
        cfg = ReinforcedV4Config()
        data = cfg.to_dict()
        assert "alignment_config" in data
        assert data["acceleration_mode"] == cfg.acceleration_mode
        assert data["alignment_beta"] == cfg.alignment_beta
        assert data["weird_operator_gain"] == cfg.weird_operator_gain
        assert data["stable_operator_gain"] == cfg.stable_operator_gain


class TestRelationalReinforcedV4:
    def test_weirdness_schedule_is_monotone(self):
        output_dir = (Path.cwd() / "data" / "retrodiction" / "_test_v4_schedule").resolve()
        try:
            shutil.rmtree(output_dir, ignore_errors=True)
            cfg = ReinforcedV4Config()
            family_ref = extract_family_inventory("latin", _latinish_sequences(), cfg.alignment_config)
            engine = RelationalReinforcedRetrodictionEngineV4(
                language="french",
                source_sequences=_tiny_sequences(),
                latin_structural_ref=_FakeStructuralRef(),
                latin_form_ref=_FakeLatinFormRef(),
                config=cfg,
                output_dir=output_dir,
                references=_FakeReferenceSet(),
                family_reference_inventory=family_ref,
            )
            assert engine._weirdness_from_alignment(0.2) > engine._weirdness_from_alignment(0.8)
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)

    def test_scheduled_weights_shift_toward_weird_ops_when_alignment_is_low(self):
        output_dir = (Path.cwd() / "data" / "retrodiction" / "_test_v4_weights").resolve()
        try:
            shutil.rmtree(output_dir, ignore_errors=True)
            cfg = ReinforcedV4Config()
            family_ref = extract_family_inventory("latin", _latinish_sequences(), cfg.alignment_config)
            engine = RelationalReinforcedRetrodictionEngineV4(
                language="french",
                source_sequences=_tiny_sequences(),
                latin_structural_ref=_FakeStructuralRef(),
                latin_form_ref=_FakeLatinFormRef(),
                config=cfg,
                output_dir=output_dir,
                references=_FakeReferenceSet(),
                family_reference_inventory=family_ref,
            )
            low = engine._scheduled_operator_weights(0.2)
            high = engine._scheduled_operator_weights(0.8)
            weird_idxs = [i for i, op in enumerate(engine._selection_schedule_dict(low).keys()) if op in {"sequence_span_rewrite", "function_word_burst", "paradigm_family_rewrite", "macro_bundle_rewrite"}]
            assert sum(low[i] for i in weird_idxs) > sum(high[i] for i in weird_idxs)
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)

    def test_run_records_alignment_diagnostics(self):
        output_dir = (Path.cwd() / "data" / "retrodiction" / "_test_v4_smoke").resolve()
        try:
            shutil.rmtree(output_dir, ignore_errors=True)
            cfg = ReinforcedV4Config(
                num_sequences=60,
                max_proposals=16,
                max_accepted_stages=6,
                patience=6,
                seed=2,
                n_candidates=4,
                min_improvement=0.00005,
                use_incremental_scoring=False,
            )
            family_ref = extract_family_inventory("latin", _latinish_sequences(), cfg.alignment_config)
            engine = RelationalReinforcedRetrodictionEngineV4(
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
            assert "family_alignment_score" in records[0].diagnostics
            assert "weirdness_level" in records[0].diagnostics

            with (output_dir / "run_summary.json").open(encoding="utf-8") as fh:
                summary = json.load(fh)
            assert summary["algorithm"] == "relational_v4"
            assert "final_family_alignment_score" in summary
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)

    def test_run_supports_batch_guidance_mode(self):
        output_dir = (Path.cwd() / "data" / "retrodiction" / "_test_v4_guided").resolve()
        try:
            shutil.rmtree(output_dir, ignore_errors=True)
            cfg = ReinforcedV4Config(
                num_sequences=60,
                max_proposals=12,
                max_accepted_stages=5,
                patience=5,
                seed=5,
                n_candidates=3,
                min_improvement=0.00005,
                acceleration_mode="numpy_batch",
                use_incremental_scoring=False,
            )
            family_ref = extract_family_inventory("latin", _latinish_sequences(), cfg.alignment_config)
            guidance_builder = _FakeGuidanceBuilder()
            engine = RelationalReinforcedRetrodictionEngineV4(
                language="french",
                source_sequences=_tiny_sequences(),
                latin_structural_ref=_FakeStructuralRef(),
                latin_form_ref=_FakeLatinFormRef(),
                config=cfg,
                output_dir=output_dir,
                references=_FakeReferenceSet(),
                family_reference_inventory=family_ref,
                batch_guidance_builder=guidance_builder,
            )

            records = engine.run()
            assert len(records) >= 1
            latest = records[-1]
            assert "batch_guidance_backend" in latest.diagnostics
            assert latest.diagnostics["batch_guidance_backend"] in {"numpy", "python_only"}
            assert records[0].diagnostics["batch_guidance_tensor_state_update_mode"] == "seed_build"
            assert guidance_builder.initial_state_builds == 1
            assert guidance_builder.build_from_state_calls >= 1
            assert guidance_builder.build_calls == 0
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)

    def test_live_event_mode_off_skips_event_file(self):
        output_dir = (Path.cwd() / "data" / "retrodiction" / "_test_v4_live_events_off").resolve()
        try:
            shutil.rmtree(output_dir, ignore_errors=True)
            cfg = ReinforcedV4Config(
                num_sequences=60,
                max_proposals=8,
                max_accepted_stages=4,
                patience=4,
                seed=3,
                n_candidates=3,
                min_improvement=0.00005,
                use_incremental_scoring=False,
                live_event_mode="off",
            )
            family_ref = extract_family_inventory("latin", _latinish_sequences(), cfg.alignment_config)
            engine = RelationalReinforcedRetrodictionEngineV4(
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
            assert not (output_dir / "live_events.jsonl").exists()
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)

    def test_live_event_mode_selected_filters_out_plain_rejections(self):
        output_dir = (Path.cwd() / "data" / "retrodiction" / "_test_v4_live_events_selected").resolve()
        try:
            shutil.rmtree(output_dir, ignore_errors=True)
            cfg = ReinforcedV4Config(
                num_sequences=60,
                max_proposals=10,
                max_accepted_stages=4,
                patience=4,
                seed=4,
                n_candidates=4,
                min_improvement=0.00005,
                use_incremental_scoring=False,
                live_event_mode="selected",
                live_event_buffer_size=256,
            )
            family_ref = extract_family_inventory("latin", _latinish_sequences(), cfg.alignment_config)
            engine = RelationalReinforcedRetrodictionEngineV4(
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
            events = [
                json.loads(line)
                for line in events_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            assert events
            assert all(event["outcome"] != "rejected" for event in events)
            assert any(event["outcome"] in {"best_rejected", "stage_committed", "accepted"} for event in events)
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)

    def test_run_materializes_sparse_winner_before_save(self, monkeypatch):
        output_dir = (Path.cwd() / "data" / "retrodiction" / "_test_v4_sparse_materialize").resolve()
        try:
            shutil.rmtree(output_dir, ignore_errors=True)
            cfg = ReinforcedV4Config(
                num_sequences=60,
                max_proposals=1,
                max_accepted_stages=3,
                patience=2,
                seed=11,
                n_candidates=1,
                min_improvement=0.00001,
                use_incremental_scoring=False,
            )
            family_ref = extract_family_inventory("latin", _latinish_sequences(), cfg.alignment_config)
            engine = RelationalReinforcedRetrodictionEngineV4(
                language="french",
                source_sequences=_tiny_sequences(),
                latin_structural_ref=_FakeStructuralRef(),
                latin_form_ref=_FakeLatinFormRef(),
                config=cfg,
                output_dir=output_dir,
                references=_FakeReferenceSet(),
                family_reference_inventory=family_ref,
            )

            saved_sparse_row: dict[str, list[str]] = {}
            original_save_stage = engine._save_stage

            def fake_mutate_candidate(sequences, rng, guidance=None, precomputed_token_counts=None, precomputed_bigram_counts=None):
                changed_row = ["novum", *sequences[0][1:]]
                return (
                    MutationPayload(changed_sequences={0: changed_row}),
                    "token_char_edit",
                    "demo sparse rewrite",
                    0.2,
                )

            def fake_score_mutation_pool(*, current, mutation_pool, proposal_guidance):
                payload, operator, details, mutation_cost = mutation_pool[0]
                candidate = CandidateState(
                    sequences=current.sequences,
                    operator=operator,
                    details=details,
                    mutation_cost=mutation_cost,
                    structural_vector=np.array(current.structural_vector, copy=True),
                    latin_structural_score=current.latin_structural_score + 0.1,
                    latin_form_score=current.latin_form_score + 0.1,
                    form_details=dict(current.form_details),
                    total_score=current.total_score + 1.0,
                    scores=dict(current.scores),
                    diagnostics=dict(current.diagnostics),
                    type_token_ratio=current.type_token_ratio,
                    bigram_coverage=current.bigram_coverage,
                    trigram_coverage=current.trigram_coverage,
                    bigram_profile=dict(current.bigram_profile),
                    trigram_profile=dict(current.trigram_profile),
                )
                return [(candidate, operator, details, mutation_cost, payload)]

            def wrapped_save_stage(candidate, *args, **kwargs):
                if kwargs.get("mutation_operator") != "seed":
                    saved_sparse_row["row0"] = candidate.sequences[0]
                return original_save_stage(candidate, *args, **kwargs)

            monkeypatch.setattr(engine, "_mutate_candidate", fake_mutate_candidate)
            monkeypatch.setattr(engine, "_score_mutation_pool", fake_score_mutation_pool)
            monkeypatch.setattr(engine, "_save_stage", wrapped_save_stage)

            records = engine.run()

            assert len(records) >= 2
            assert saved_sparse_row["row0"][0] == "novum"
            assert records[-1].mutation_operator == "token_char_edit"
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)

    def test_sequence_span_sparse_returns_sparse_payload(self):
        output_dir = (Path.cwd() / "data" / "retrodiction" / "_test_v4_span_sparse").resolve()
        try:
            shutil.rmtree(output_dir, ignore_errors=True)
            cfg = ReinforcedV4Config(
                num_sequences=60,
                max_proposals=1,
                max_accepted_stages=3,
                patience=2,
                seed=13,
                n_candidates=1,
                min_improvement=0.00001,
                use_incremental_scoring=False,
            )
            family_ref = extract_family_inventory("latin", _latinish_sequences(), cfg.alignment_config)
            engine = RelationalReinforcedRetrodictionEngineV4(
                language="french",
                source_sequences=_tiny_sequences(),
                latin_structural_ref=_FakeStructuralRef(),
                latin_form_ref=_FakeLatinFormRef(),
                config=cfg,
                output_dir=output_dir,
                references=_FakeReferenceSet(),
                family_reference_inventory=family_ref,
            )

            payload = None
            for seed in range(20):
                payload, details, cost = engine._mutate_sequence_span_rewrite_sparse(
                    _tiny_sequences(),
                    np.random.default_rng(seed),
                )
                if payload is not None:
                    break

            assert payload is not None
            assert payload.is_sparse
            assert payload.changed_sequences
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)

    def test_guided_token_edit_payload_is_sparse(self):
        output_dir = (Path.cwd() / "data" / "retrodiction" / "_test_v4_guided_sparse").resolve()
        try:
            shutil.rmtree(output_dir, ignore_errors=True)
            cfg = ReinforcedV4Config(
                num_sequences=60,
                max_proposals=1,
                max_accepted_stages=3,
                patience=2,
                seed=17,
                n_candidates=1,
                min_improvement=0.00001,
                use_incremental_scoring=False,
            )
            family_ref = extract_family_inventory("latin", _latinish_sequences(), cfg.alignment_config)
            engine = RelationalReinforcedRetrodictionEngineV4(
                language="french",
                source_sequences=_tiny_sequences(),
                latin_structural_ref=_FakeStructuralRef(),
                latin_form_ref=_FakeLatinFormRef(),
                config=cfg,
                output_dir=output_dir,
                references=_FakeReferenceSet(),
                family_reference_inventory=family_ref,
            )
            sequences = _tiny_sequences()
            token_counts = engine._token_counts(sequences)
            bigram_counts = engine._bigram_counts(sequences)
            guidance = _FakeGuidanceBuilder()._guidance(anchor_vocab_size=5)

            payload, details, cost = engine._apply_named_operator_guided_payload(
                "token_char_edit",
                sequences,
                token_counts,
                bigram_counts,
                guidance,
                np.random.default_rng(23),
            )

            assert payload is not None
            assert payload.is_sparse
            assert payload.changed_sequences
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)

    def test_macro_bundle_sparse_returns_sparse_payload(self, monkeypatch):
        output_dir = (Path.cwd() / "data" / "retrodiction" / "_test_v4_macro_sparse").resolve()
        try:
            shutil.rmtree(output_dir, ignore_errors=True)
            cfg = ReinforcedV4Config(
                num_sequences=60,
                max_proposals=1,
                max_accepted_stages=3,
                patience=2,
                seed=19,
                n_candidates=1,
                min_improvement=0.00001,
                use_incremental_scoring=False,
                macro_bundle_min_steps=1,
                macro_bundle_max_steps=1,
            )
            family_ref = extract_family_inventory("latin", _latinish_sequences(), cfg.alignment_config)
            engine = RelationalReinforcedRetrodictionEngineV4(
                language="french",
                source_sequences=_tiny_sequences(),
                latin_structural_ref=_FakeStructuralRef(),
                latin_form_ref=_FakeLatinFormRef(),
                config=cfg,
                output_dir=output_dir,
                references=_FakeReferenceSet(),
                family_reference_inventory=family_ref,
            )
            sequences = _tiny_sequences()

            def fake_apply_named_operator_payload(operator, sequences, token_counts, bigram_counts, rng):
                return (
                    MutationPayload(changed_sequences={0: ["macro", *sequences[0][1:]]}),
                    "macro demo",
                    0.25,
                )

            monkeypatch.setattr(engine, "_apply_named_operator_payload", fake_apply_named_operator_payload)

            payload, details, cost = engine._mutate_macro_bundle_rewrite_sparse(
                sequences,
                np.random.default_rng(29),
            )

            assert payload is not None
            assert payload.is_sparse
            assert payload.changed_sequences is not None
            assert payload.changed_sequences[0][0] == "macro"
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)

    def test_macro_bundle_materialized_wrapper_uses_payload(self, monkeypatch):
        output_dir = (Path.cwd() / "data" / "retrodiction" / "_test_v4_macro_materialized").resolve()
        try:
            shutil.rmtree(output_dir, ignore_errors=True)
            cfg = ReinforcedV4Config(
                num_sequences=60,
                max_proposals=1,
                max_accepted_stages=3,
                patience=2,
                seed=21,
                n_candidates=1,
                min_improvement=0.00001,
                use_incremental_scoring=False,
            )
            family_ref = extract_family_inventory("latin", _latinish_sequences(), cfg.alignment_config)
            engine = RelationalReinforcedRetrodictionEngineV4(
                language="french",
                source_sequences=_tiny_sequences(),
                latin_structural_ref=_FakeStructuralRef(),
                latin_form_ref=_FakeLatinFormRef(),
                config=cfg,
                output_dir=output_dir,
                references=_FakeReferenceSet(),
                family_reference_inventory=family_ref,
            )
            sequences = _tiny_sequences()

            def fake_macro_payload(sequences, rng, guidance=None):
                return (
                    MutationPayload(changed_sequences={0: ["macrox", *sequences[0][1:]]}),
                    "macro payload",
                    0.9,
                )

            monkeypatch.setattr(engine, "_mutate_macro_bundle_rewrite_payload", fake_macro_payload)

            mutated, details, cost = engine._mutate_macro_bundle_rewrite(
                sequences,
                np.random.default_rng(31),
            )

            assert mutated is not None
            assert mutated[0][0] == "macrox"
            assert details == "macro payload"
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)
