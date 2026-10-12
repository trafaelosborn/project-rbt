"""Registration, data-selection, and readout exclusion checks for Decision 047."""
from copy import deepcopy
from dataclasses import replace
import json

import numpy as np
import pytest

from src.control import h1_ablation as launch
from src.retrodiction.h1_checkpointed import SearchConfig, atomic_json, read_json, seal
from src.retrodiction.h1_objective import text_hash
from src.validation import h1_ablation as readout


@pytest.fixture
def cheap_plan(monkeypatch):
    source = {"schema": "h1_ablation_registration_v1", "purpose": "production", "root": "unused",
              "identity": {"protocol": read_json(launch.PROJECT_ROOT / launch.PROTOCOL),
                           "code": "v1"}, "jobs": [{"job_id": "french"}]}
    def make(root, purpose="production"):
        return {**deepcopy(source), "root": str(root.resolve()), "purpose": purpose}
    monkeypatch.setattr(launch, "make_plan", make)
    return source


def test_registration_requires_new_root_and_rejects_changed_identity(tmp_path, monkeypatch, cheap_plan):
    root = tmp_path / "new"
    original = launch.register(root)
    assert launch.validate_registration(root) == original
    with pytest.raises(FileExistsError):
        launch.register(root)
    cheap_plan["identity"]["code"] = "v2"
    with pytest.raises(ValueError, match="changed"):
        launch.validate_registration(root)


def test_registration_digest_tamper_rejected(tmp_path, cheap_plan):
    root = tmp_path / "new"
    registration = launch.register(root)
    registration["purpose"] = "pilot"
    atomic_json(root / launch.SIDECAR, registration)
    with pytest.raises(ValueError, match="integrity"):
        launch.validate_registration(root)


def test_pilot_readout_excluded_before_loading_runs(tmp_path, cheap_plan):
    root = tmp_path / "pilot"
    launch.register(root, "pilot")
    with pytest.raises(ValueError, match="Pilot registrations"):
        readout.analyze(root)


def test_actual_protocol_pins_and_matrix(tmp_path):
    for path, digest in launch.PINNED.items():
        assert text_hash(launch.PROJECT_ROOT / path) == digest
    jobs = launch.jobs(tmp_path, "production")
    assert len(jobs) == 66
    assert len({j["job_id"] for j in jobs}) == 66
    for language in {j["search_config"]["language"] for j in jobs}:
        same = [j for j in jobs if j["search_config"]["language"] == language]
        assert len({j["sampled_corpus_sha256"] for j in same}) == 1
        assert {j["search_config"]["search_seed"] for j in same} == {42, 1729, 8675309}
        assert {j["search_config"]["form_reward_mode"] for j in same} == {"full", "disabled"}
    assert all(j["search_config"]["proposal_budget"] == 1000 for j in jobs)
    assert all(j["engine_config"]["enable_culture_bombs"] is False for j in jobs)


def test_corpus_seed_is_independent_of_search_seed():
    first = SearchConfig("french", corpus_seed=42, search_seed=42)
    second = replace(first, search_seed=1729)
    third = replace(first, corpus_seed=1729)
    a, b, c = (launch.sampled_source(cfg) for cfg in (first, second, third))
    assert a == b
    assert a != c
    assert all(sum(map(len, sequences)) == 20000 for sequences in (a, b, c))


def test_incomplete_main_run_fails_and_missing_stress_is_separate(tmp_path, monkeypatch):
    protocol = read_json(launch.PROJECT_ROOT / launch.PROTOCOL)
    jobs = []
    for language in protocol["romance"] + protocol["controls"] + protocol["stress_sources"]:
        for seed in protocol["search_seeds"]:
            for arm in protocol["form_reward_modes"]:
                cfg = SearchConfig(language, search_seed=seed, form_reward_mode=arm)
                job_id = f"{language}_{seed}_{arm}"
                jobs.append({"job_id": job_id, "search_config": cfg.to_dict()})
                if language not in protocol["stress_sources"]:
                    (tmp_path / job_id).mkdir()
                    (tmp_path / job_id / "checkpoint.json").touch()
    registration = {"purpose": "production", "sha256": "test", "identity": {"protocol": protocol}, "jobs": jobs}
    monkeypatch.setattr(launch, "validate_registration", lambda root: registration)
    monkeypatch.setattr(launch, "job_contract", lambda *args: {})
    monkeypatch.setattr(launch, "make_engine", lambda *args: None)
    complete = False
    def audit(engine, cfg, output, contract):
        index = (protocol["romance"] + protocol["controls"]).index(cfg.language)
        return {"complete": complete, "checkpoints": {str(p): {"measurement": {"z_6d": [index, 0, 0, 0, 0, 0]}}
                                                     for p in cfg.proposal_checkpoints}}
    monkeypatch.setattr(readout, "audit_search", audit)
    with pytest.raises(ValueError, match="incomplete"):
        readout.analyze(tmp_path)
    complete = True
    result = readout.analyze(tmp_path)
    assert result["status"] == "descriptive_complete"
    assert len(result["main_diagnostics"]) == 54
    assert len(result["unavailable_stress_runs"]) == 12
    assert result["statistics"]["mean_full_minus_disabled_auc"] == 0


def test_configuration_roundtrip_rejects_changed_policy():
    cfg = SearchConfig("french")
    assert SearchConfig.from_dict(cfg.to_dict()) == cfg
    changed = {**cfg.to_dict(), "rng_policy": "restart_each_block"}
    with pytest.raises(ValueError, match="policy mismatch"):
        SearchConfig.from_dict(changed)
