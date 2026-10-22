"""Operational queue behavior; scientific engine settings are immutable."""
from pathlib import Path
import subprocess

import pytest

from scripts import run_h1_ablation_matrix as runner
from src.retrodiction.h1_checkpointed import atomic_json, seal
from src.retrodiction.h1_objective import stable_hash


def fixture(monkeypatch, tmp_path, *, fail=None, purpose="pilot", access_error=False, failures=None):
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
            self.fails = self.name == fail and (failures is None or calls.count(self.name) <= failures)
            if self.fails and access_error:
                checkpoint = str((tmp_path / self.name / "checkpoint.json").resolve())
                kwargs["stdout"].write(f"PermissionError: [Errno 13] Permission denied: {checkpoint!r}\n")
        def poll(self):
            if self.fails:
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


def test_access_failure_is_audited_before_retry_and_log_is_retained(tmp_path, monkeypatch):
    _, calls = fixture(monkeypatch, tmp_path, fail="a", access_error=True, failures=1)
    monkeypatch.setattr(runner.h1_io, "WINDOWS", True)
    # The OS-independent queue fixture has no scientific history. Real-history
    # recovery and corrupted history are tested separately below.
    audits = []
    def audit(root, registration, job, **kwargs):
        audits.append(job["job_id"])
        return {"sha256": "verified-fixture-checkpoint", "proposal_index": 0}
    monkeypatch.setattr(runner, "recovery_checkpoint", audit)
    result = runner.run_matrix(tmp_path, poll_seconds=.01, minimum_free_gib=0)
    assert result["status"] == "complete" and audits == ["a"]
    assert result["worker_recoveries"]["a"] == 1
    events = result["events"]
    recovery = next(e for e in events if e["event"] == "checkpoint_access_recovery")
    assert Path(recovery["failed_log"]).read_text().startswith("PermissionError:")
    assert calls.count("a") == 3  # failed attempt, then two ordinary sessions


def test_persistent_worker_access_failure_has_a_fixed_recovery_cap(tmp_path, monkeypatch):
    _, calls = fixture(monkeypatch, tmp_path, fail="a", access_error=True)
    monkeypatch.setattr(runner.h1_io, "WINDOWS", True)
    monkeypatch.setattr(runner, "recovery_checkpoint", lambda *a, **k: {"sha256": "fixture", "proposal_index": 0})
    result = runner.run_matrix(tmp_path, poll_seconds=.01, minimum_free_gib=0)
    assert result["status"] == "failed"
    assert result["reason"] == "child_failed:a:exit_1"
    assert calls.count("a") == 4 and result["worker_recoveries"]["a"] == 3


def test_failed_recovery_audit_never_relaunches_worker(tmp_path, monkeypatch):
    _, calls = fixture(monkeypatch, tmp_path, fail="a", access_error=True, failures=1)
    monkeypatch.setattr(runner.h1_io, "WINDOWS", True)
    def corrupt(*args, **kwargs):
        raise ValueError("Proposal chain is discontinuous")
    monkeypatch.setattr(runner, "recovery_checkpoint", corrupt)
    with pytest.raises(ValueError, match="discontinuous"):
        runner.run_matrix(tmp_path, poll_seconds=.01, minimum_free_gib=0)
    result = runner.h1_io.read_json(tmp_path / "matrix_status.json")
    assert calls.count("a") == 1 and result["status"] == "failed"
    assert result["worker_recoveries"]["a"] == 0 and not result["active"]


def test_exception_drain_refreshes_final_committed_progress(tmp_path, monkeypatch):
    fixture(monkeypatch, tmp_path)
    actual = runner.progress
    fired = []
    def fail_once(root, registration, job, **kwargs):
        if (root / job["job_id"] / "checkpoint.json").exists() and not fired:
            fired.append(True)
            raise PermissionError(13, "persistent checkpoint access error after retry budget")
        return actual(root, registration, job, **kwargs)
    monkeypatch.setattr(runner, "progress", fail_once)
    with pytest.raises(PermissionError):
        runner.run_matrix(tmp_path, poll_seconds=.01, minimum_free_gib=0)
    result = runner.h1_io.read_json(tmp_path / "matrix_status.json")
    assert result["status"] == "failed" and not result["active"]
    assert result["progress"] == {"a": 2, "b": 2, "c": 0}
    assert len([e for e in result["events"] if e["event"] == "drained"]) == 2


def test_status_write_failure_still_drains_started_worker(tmp_path, monkeypatch):
    _, calls = fixture(monkeypatch, tmp_path)
    actual = runner.atomic_json
    def broken(path, value):
        if value["status"] in {"running", "stopping"}:
            raise OSError(28, "disk full")
        return actual(path, value)
    monkeypatch.setattr(runner, "atomic_json", broken)
    with pytest.raises(OSError):
        runner.run_matrix(tmp_path, poll_seconds=.01, minimum_free_gib=0)
    result = runner.h1_io.read_json(tmp_path / "matrix_status.json")
    assert result["status"] == "failed" and not result["active"]
    assert calls == ["a"] and result["progress"]["a"] == 2
    assert result["shutdown_errors"][0]["operation"] == "publish_stopping"


def test_queue_does_not_open_checkpoints_while_writer_holds_job_lock(tmp_path, monkeypatch):
    registration, _ = fixture(monkeypatch, tmp_path)
    polls = []
    class Child:
        def __init__(self, command, **kwargs):
            self.name = command[command.index("--job-id") + 1]
            self.pid = 1234
            self.lock = runner.run_lock(tmp_path / self.name / "run.lock")
            self.lock.__enter__()
            self.done = False
            self.count = 0
        def poll(self):
            self.count += 1
            polls.append((self.name, self.count))
            if self.count < 3:
                return None
            if not self.done:
                job = next(j for j in registration["jobs"] if j["job_id"] == self.name)
                atomic_json(tmp_path / self.name / "checkpoint.json", seal({
                    "contract_sha256": stable_hash(runner.job_contract(registration, job)),
                    "proposal_index": 4}))
                self.lock.__exit__(None, None, None)
                self.done = True
            return 0
        def wait(self):
            while self.poll() is None:
                pass
            return 0
    monkeypatch.setattr(runner.subprocess, "Popen", Child)
    result = runner.run_matrix(tmp_path, poll_seconds=.01, minimum_free_gib=0)
    assert result["status"] == "complete" and result["completed_jobs"] == 3
    assert len(polls) == 9


def test_progress_refuses_to_read_an_existing_live_writer(tmp_path, monkeypatch):
    registration, _ = fixture(monkeypatch, tmp_path)
    folder = tmp_path / "a"
    folder.mkdir()
    with runner.run_lock(folder / "run.lock"):
        with pytest.raises(ValueError, match="Another process"):
            runner.progress(tmp_path, registration, registration["jobs"][0])
