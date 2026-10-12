"""Behavioral validation of the Decision 047 controller and reward ablation."""
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from src.retrodiction.engine_reinforced_v4 import MutationPayload
from src.retrodiction.h1_checkpointed import (
    FORM_TERMS, SearchConfig, atomic_json, audit_search, candidate_dict, diversity,
    engine_config, load_checkpoint, read_json, reward_components, run_search, seal,
)
from src.retrodiction.h1_objective import stable_hash
from src.retrodiction.h1_run_lock import run_lock
from src.validation.h1_ablation import panel_statistics, render_report
from tests.test_h1_execution import objective, engine_for, corpus, collapsed


def configured(tmp_path, monkeypatch, objective, config):
    engine = engine_for(tmp_path, monkeypatch, objective, use_incremental_scoring=False)
    engine.config = engine_config(config, Path("source.json"), tmp_path)
    def mutate(sequences, rng, **kwargs):
        seqs = [list(seq) for seq in sequences]
        if sequences == collapsed():
            seqs = corpus()
        else:
            row = int(rng.integers(len(seqs)))
            column = int(rng.integers(6))
            seqs[row][column] += str(rng.choice(["a", "z", "us"]))
        return MutationPayload(sequences=seqs), "test_mutation", "deterministic given state and RNG", 0.3
    monkeypatch.setattr(engine, "_mutate_candidate", mutate)
    return engine


def contract(engine, config):
    return {"search_config": config.to_dict(), "engine_config": engine.config.to_dict()}


@pytest.mark.parametrize("arm", ["full", "disabled"])
def test_pause_resume_matches_uninterrupted_every_committed_byte(tmp_path, monkeypatch, objective, arm):
    config = SearchConfig("french", 73, 1729, arm, 6, 3, (0, 2, 6))
    whole, split = tmp_path / "whole", tmp_path / "split"
    engine = configured(whole, monkeypatch, objective, config)
    identity = contract(engine, config)
    result = run_search(engine, config, whole, identity, collapsed())
    assert result["status"] == "complete"
    for target in (2, 1, 3):
        engine = configured(split, monkeypatch, objective, config)
        result = run_search(engine, config, split, identity, collapsed(), max_session_proposals=target)
    assert result["status"] == "complete"
    assert read_json(whole / "checkpoint.json") == read_json(split / "checkpoint.json")
    for directory in ("states", "proposals"):
        expected = {p.name: p.read_bytes() for p in (whole / directory).glob("*.json")}
        actual = {p.name: p.read_bytes() for p in (split / directory).glob("*.json")}
        assert actual == expected
    audit = audit_search(engine, config, split, identity)
    assert audit["states_replayed"] >= 2
    assert audit["proposals_replayed"] == 6
    assert set(audit["checkpoints"]) == {"0", "2", "6"}


def test_interrupt_between_event_and_pointer_replays_identically(tmp_path, monkeypatch, objective):
    import src.retrodiction.h1_checkpointed as module
    cfg = SearchConfig("french", proposal_budget=3, n_candidates=2, proposal_checkpoints=(0, 3))
    engine = configured(tmp_path, monkeypatch, objective, cfg)
    identity = contract(engine, cfg)
    real_write = module.atomic_json
    def interrupted(path, value):
        if path.name == "checkpoint.json" and value["proposal_index"] == 2:
            raise KeyboardInterrupt("simulated power loss before pointer publication")
        real_write(path, value)
    monkeypatch.setattr(module, "atomic_json", interrupted)
    with pytest.raises(KeyboardInterrupt):
        run_search(engine, cfg, tmp_path, identity, collapsed())
    assert read_json(tmp_path / "checkpoint.json")["proposal_index"] == 1
    orphan = (tmp_path / "proposals" / "000000002.json").read_bytes()
    monkeypatch.setattr(module, "atomic_json", real_write)
    engine = configured(tmp_path, monkeypatch, objective, cfg)
    result = run_search(engine, cfg, tmp_path, identity, collapsed())
    assert result["status"] == "complete"
    assert (tmp_path / "proposals" / "000000002.json").read_bytes() == orphan
    audit_search(engine, cfg, tmp_path, identity)


def test_ablation_removes_every_form_reward_and_retains_other_terms(tmp_path, monkeypatch, objective):
    full_cfg = SearchConfig("french", proposal_budget=1, proposal_checkpoints=(0, 1))
    full = configured(tmp_path / "full", monkeypatch, objective, full_cfg)
    off_cfg = replace(full_cfg, form_reward_mode="disabled")
    off = configured(tmp_path / "off", monkeypatch, objective, off_cfg)
    changed = {k for k, v in full.config.to_dict().items() if off.config.to_dict()[k] != v}
    assert changed == set(FORM_TERMS)
    for engine in (full, off):
        before = engine._evaluate_sequences(collapsed(), 0)
        after = engine._evaluate_sequences(corpus(), 1)
        # Exercise all gain and conditional bonus branches independently of
        # the fixture form reference's score ordering.
        before.latin_form_score = 0
        before.form_details = {"latin_suffix_cosine": 0, "latin_char_trigram_cosine": 0}
        after.latin_form_score = 1
        after.form_details = {"latin_suffix_cosine": 1, "latin_char_trigram_cosine": 1}
        after.total_score += engine.config.form_weight * (1 - after.diagnostics["h1"]["score_components"]["form"] / max(engine.config.form_weight, 1e-100))
        after.diagnostics["h1"]["score_components"]["form"] = engine.config.form_weight
        after = engine._amplify_reward(before, after)
        parts = reward_components(engine, before, after)
        terms = [parts[k] for k in ("form", "form_gain_reward", "suffix_gain_reward", "trigram_gain_reward", "joint_reward", "cost_relief")]
        assert all(v > 0 for v in terms) if engine is full else all(v == 0 for v in terms)
        assert parts["struct_gain_reward"] > 0
        assert parts["mutation_cost"] < 0


def test_candidate_counters_partition_slots(tmp_path, monkeypatch, objective):
    cfg = SearchConfig("french", proposal_budget=1, n_candidates=4, proposal_checkpoints=(0, 1))
    engine = configured(tmp_path, monkeypatch, objective, cfg)
    attempts = iter([(None, "none", "no payload", 0),
                     (MutationPayload(sequences=corpus()[:-1]), "merge", "short", 0),
                     (MutationPayload(sequences=corpus()), "good", "improved", 0),
                     (MutationPayload(sequences=collapsed()), "unchanged", "same", 0)])
    monkeypatch.setattr(engine, "_mutate_candidate", lambda *a, **k: next(attempts))
    identity = contract(engine, cfg)
    run_search(engine, cfg, tmp_path, identity, collapsed())
    event = read_json(tmp_path / "proposals" / "000000001.json")
    assert [a["outcome"] for a in event["attempts"]] == ["no_op", "invalid_token_budget", "accepted", "rejected"]
    assert event["attempts"][-1]["unchanged_corpus"] is True
    assert sum(sum(v.values()) for v in event["counters"].values()) == 4
    assert audit_search(engine, cfg, tmp_path, identity)["evaluated_candidates"] == 2


def test_pause_file_and_time_limit_are_not_completion(tmp_path, monkeypatch, objective):
    import src.retrodiction.h1_checkpointed as module
    cfg = SearchConfig("french", proposal_budget=2, n_candidates=1, proposal_checkpoints=(0, 2))
    engine = configured(tmp_path, monkeypatch, objective, cfg)
    identity = contract(engine, cfg)
    (tmp_path / "PAUSE").touch()
    result = run_search(engine, cfg, tmp_path, identity, collapsed())
    assert result["status"] == "paused" and result["scientific_termination"] is None
    (tmp_path / "PAUSE").unlink()
    ticks = iter([0, 10, 10])
    monkeypatch.setattr(module.time, "monotonic", lambda: next(ticks))
    result = run_search(engine, cfg, tmp_path, identity, collapsed(), max_session_seconds=5)
    assert result["reason"] == "session_time_limit" and result["proposal_index"] == 0


def test_altered_config_and_saved_state_rejected(tmp_path, monkeypatch, objective):
    cfg = SearchConfig("french", proposal_budget=2, n_candidates=1, proposal_checkpoints=(0, 2))
    engine = configured(tmp_path, monkeypatch, objective, cfg)
    identity = contract(engine, cfg)
    run_search(engine, cfg, tmp_path, identity, collapsed(), max_session_proposals=1)
    with pytest.raises(ValueError, match="configuration"):
        run_search(engine, replace(cfg, search_seed=7), tmp_path, identity, collapsed())
    checkpoint = read_json(tmp_path / "checkpoint.json")
    state_path = tmp_path / "states" / (checkpoint["state_sha256"] + ".json")
    state = read_json(state_path)
    state["candidate"]["total_score"] += 1
    atomic_json(state_path, state)
    with pytest.raises(ValueError, match="integrity"):
        run_search(engine, cfg, tmp_path, identity, collapsed())


def test_exclusive_writer_lock_released_after_exception(tmp_path):
    path = tmp_path / "lock"
    with run_lock(path):
        with pytest.raises(ValueError, match="Another process"):
            with run_lock(path):
                pass
    with run_lock(path):
        pass


@pytest.mark.parametrize("change", [
    {"corpus_seed": -1}, {"search_seed": True}, {"form_reward_mode": "unknown"},
    {"proposal_budget": 0}, {"n_candidates": 0}, {"proposal_checkpoints": (0, 1000, 1000)},
    {"proposal_checkpoints": (1, 1000)}, {"proposal_checkpoints": (0, 1)},
])
def test_bad_config_rejected(change):
    with pytest.raises(ValueError):
        replace(SearchConfig("french"), **change).validate()


def test_diversity_known_corpora():
    assert diversity([["x"] * 8])["word_entropy_bits"] == 0
    values = diversity([["a", "b", "a", "b"]])
    assert values["word_entropy_bits"] == 1
    assert values["maximum_token_frequency"] == 0.5
    assert values["vocabulary_size"] == 2


def panel_fixture():
    protocol = {"romance": ["a", "b", "c", "d", "e"], "controls": ["f", "g", "h", "i"],
                "search_seeds": [42, 1729, 8675309], "form_reward_modes": ["full", "disabled"],
                "proposal_checkpoints": [0, 1, 2], "proposal_budget": 2}
    trajectories = {}
    for seed in protocol["search_seeds"]:
        for arm in protocol["form_reward_modes"]:
            for index, language in enumerate(protocol["romance"] + protocol["controls"]):
                start = np.array([index + 1., 0, 0, 0, 0, 0])
                end = start * (0.1 if arm == "full" and index < 5 else 1)
                trajectories[(language, seed, arm)] = np.array([start, (start + end) / 2, end])
    return protocol, trajectories


def test_panel_known_direction_zero_disabled_and_seed_duplication():
    protocol, trajectories = panel_fixture()
    result = panel_statistics(trajectories, protocol)
    assert result["mean_full_minus_disabled_auc"] > 0
    assert len(result["pair_records"]) == 216
    assert all(row["arms"]["disabled"]["proposal_budget_auc_contrast"] == 0 for row in result["per_seed"])
    one = panel_statistics(trajectories, {**protocol, "search_seeds": [42]})
    assert one["mean_full_minus_disabled_auc"] == pytest.approx(result["mean_full_minus_disabled_auc"])


def test_panel_missing_or_degenerate_is_not_evaluable():
    protocol, trajectories = panel_fixture()
    del trajectories[("a", 42, "full")]
    with pytest.raises(ValueError, match="Missing"):
        panel_statistics(trajectories, protocol)
    protocol, trajectories = panel_fixture()
    trajectories[("a", 42, "full")] = trajectories[("b", 42, "full")]
    with pytest.raises(ValueError, match="Zero initial"):
        panel_statistics(trajectories, protocol)
    assert "No scientific verdict" in render_report({"status": "not_evaluable", "reason": "missing"})
