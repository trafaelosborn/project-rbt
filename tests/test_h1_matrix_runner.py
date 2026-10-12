"""Operational queue behavior; scientific engine settings are immutable."""
from pathlib import Path
import subprocess

import pytest

from scripts import run_h1_ablation_matrix as runner
from src.retrodiction.h1_checkpointed import atomic_json, seal
from src.retrodiction.h1_objective import stable_hash


def fixture(monkeypatch, tmp_path, *, fail=None, purpose="pilot"):
    registration = {"sha256": "test", "identity": {}, "purpose": purpose,
                    "jobs": [{"job_id": name, "search_config": {"proposal_budget": 4}}
                             for name in ("a", "b", "c")]}
    monkeypatch.setattr(runner, "validate_registration", lambda root: registration)
    monkeypatch.setattr(runner.time, "sleep", lambda seconds: None)
    calls = []
    positions = {name: 0 for name in ("a", "b", "c")}
    class Child:
        def __init__(self, command, **kwargs):
            self.name = command[command.index("--job-id") + 1]
            self.pid = 1000 + len(calls)
            calls.append(self.name)
            self.done = False
        def poll(self):
            if self.name == fail:
                return 1
            if not self.done:
                positions[self.name] = min(positions[self.name] + 2, 4)
                job = next(j for j in registration["jobs"] if j["job_id"] == self.name)
                atomic_json(tmp_path / self.name / "checkpoint.json", seal({
                    "contract_sha256": stable_hash(runner.job_contract(registration, job)),
                    "proposal_index": positions[self.name]}))
                self.done = True
            return 0
        def wait(self):
            return self.poll()
    monkeypatch.setattr(runner.subprocess, "Popen", Child)
    return registration, calls


def test_rotation_completes_all_jobs_without_changing_settings(tmp_path, monkeypatch):
    _, calls = fixture(monkeypatch, tmp_path)
    result = runner.run_matrix(tmp_path, workers=2, session_proposals=2, poll_seconds=.01, minimum_free_gib=0)
    assert result["status"] == "complete"
    assert result["completed_jobs"] == 3
    assert calls == ["a", "b", "c", "a", "b", "c"]
    command = result["events"][0]["command"]
    assert "--max-session-proposals" in command
    assert "--n-candidates" not in command


def test_failure_stops_queue_and_preserves_other_checkpoint(tmp_path, monkeypatch):
    _, calls = fixture(monkeypatch, tmp_path, fail="a")
    result = runner.run_matrix(tmp_path, workers=2, poll_seconds=.01, minimum_free_gib=0)
    assert result["status"] == "failed"
    assert calls == ["a", "b"]
    assert (tmp_path / "b/PAUSE").exists()
    assert result["progress"]["b"] == 2
    assert result["reason"] == "child_failed:a:exit_1"


def test_global_pause_launches_nothing(tmp_path, monkeypatch):
    _, calls = fixture(monkeypatch, tmp_path)
    (tmp_path / "PAUSE").touch()
    result = runner.run_matrix(tmp_path, poll_seconds=.01, minimum_free_gib=0)
    assert result["status"] == "paused"
    assert calls == []


def test_production_runs_analysis_after_all_jobs_complete(tmp_path, monkeypatch):
    _, calls = fixture(monkeypatch, tmp_path, purpose="production")
    analyses = []
    def analyze(command, **kwargs):
        analyses.append(command)
        return subprocess.CompletedProcess(command, 0)
    monkeypatch.setattr(runner.subprocess, "run", analyze)
    result = runner.run_matrix(tmp_path, poll_seconds=.01, minimum_free_gib=0)
    assert result["status"] == "complete"
    assert len(analyses) == 1
    assert "src.validation.h1_ablation" in analyses[0]


def test_low_disk_pauses_before_launch(tmp_path, monkeypatch):
    _, calls = fixture(monkeypatch, tmp_path)
    result = runner.run_matrix(tmp_path, minimum_free_gib=1e10)
    assert result["status"] == "paused"
    assert result["reason"] == "disk_free_below_operational_floor"
    assert not calls


def test_already_completed_queue_does_not_relaunch(tmp_path, monkeypatch):
    _, calls = fixture(monkeypatch, tmp_path)
    runner.run_matrix(tmp_path, poll_seconds=.01, minimum_free_gib=0)
    calls.clear()
    result = runner.run_matrix(tmp_path, poll_seconds=.01, minimum_free_gib=0)
    assert result["status"] == "complete"
    assert calls == []
