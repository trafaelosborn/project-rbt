"""The unchanged scientific engine exactly replays checkpoint-write failures."""
from pathlib import Path

import pytest

from scripts import run_h1_ablation_matrix as runner
from src.retrodiction import h1_checkpointed as science
from tests.test_h1_checkpointed import configured, contract
from tests.test_h1_execution import objective, collapsed


@pytest.mark.parametrize("arm", ["full", "disabled"])
def test_access_failure_recovery_preserves_every_scientific_byte(tmp_path, monkeypatch, objective, arm):
    cfg = science.SearchConfig("french", 42, 1729, arm, 3, 2, (0, 3))
    whole, split = tmp_path / "whole", tmp_path / "job"
    engine = configured(whole, monkeypatch, objective, cfg)
    job = {"job_id": "job", **contract(engine, cfg)}
    registration = {"sha256": "fixture", "identity": {}, "purpose": "pilot"}
    identity = runner.job_contract(registration, job)
    science.run_search(engine, cfg, whole, identity, collapsed())
    real_write = science.atomic_json
    def denied(path, value):
        if Path(path).name == "checkpoint.json" and value["proposal_index"] == 2:
            raise PermissionError(13, "simulated Windows checkpoint replacement denial", str(path))
        return real_write(path, value)
    monkeypatch.setattr(science, "atomic_json", denied)
    engine = configured(split, monkeypatch, objective, cfg)
    with pytest.raises(PermissionError):
        science.run_search(engine, cfg, split, identity, collapsed())
    orphan = (split / "proposals/000000002.json").read_bytes()
    monkeypatch.setattr(science, "atomic_json", real_write)
    verified = runner.recovery_checkpoint(tmp_path, registration, job)
    assert verified["proposal_index"] == 1
    engine = configured(split, monkeypatch, objective, cfg)
    result = science.run_search(engine, cfg, split, identity, collapsed())
    assert result["status"] == "complete"
    assert (split / "proposals/000000002.json").read_bytes() == orphan
    for directory in ("states", "proposals"):
        assert {p.name: p.read_bytes() for p in (split / directory).glob("*.json")} == {
            p.name: p.read_bytes() for p in (whole / directory).glob("*.json")}
    assert (whole / "checkpoint.json").read_bytes() == (split / "checkpoint.json").read_bytes()
    science.audit_search(engine, cfg, split, identity)


@pytest.mark.parametrize("corruption", ["checkpoint", "history", "state"])
def test_recovery_check_rejects_corrupted_evidence(tmp_path, monkeypatch, objective, corruption):
    folder = tmp_path / "job"
    cfg = science.SearchConfig("french", proposal_budget=1, n_candidates=1, proposal_checkpoints=(0, 1))
    engine = configured(folder, monkeypatch, objective, cfg)
    job = {"job_id": "job", **contract(engine, cfg)}
    registration = {"sha256": "fixture", "identity": {}, "purpose": "pilot"}
    science.run_search(engine, cfg, folder, runner.job_contract(registration, job), collapsed())
    checkpoint = science.read_json(folder / "checkpoint.json")
    path = {"checkpoint": folder / "checkpoint.json", "history": folder / "proposals/000000000.json",
            "state": folder / "states" / (checkpoint["state_sha256"] + ".json")}[corruption]
    original = science.read_json(path)
    original["tampered"] = True
    science.atomic_json(path, original)
    with pytest.raises(ValueError, match="integrity"):
        runner.recovery_checkpoint(tmp_path, registration, job)
