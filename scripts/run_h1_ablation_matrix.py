"""Operational queue for an already registered H1 ablation matrix.

The frozen engine/protocol are unchanged. This wrapper only chooses which
registered job runs next and records process/log/queue state.
"""
from __future__ import annotations
import argparse
from collections import deque
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.control.h1_ablation import validate_registration, job_contract
from src.control import h1_io
from src.retrodiction.h1_checkpointed import atomic_json, verify, load_checkpoint, load_state
from src.retrodiction.h1_objective import stable_hash, text_hash
from src.retrodiction.h1_run_lock import run_lock


def now():
    return datetime.now(timezone.utc).isoformat()


def progress(root, registration, job, *, on_retry=None):
    path = root / job["job_id"] / "checkpoint.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    # Never open a pointer concurrently with its registered writer, including
    # an orphan worker left behind by a different runner process.
    with run_lock(path.parent / "run.lock"):
        try:
            checkpoint = verify(h1_io.read_json(path, on_retry=on_retry))
        except FileNotFoundError:
            return 0
    if checkpoint["contract_sha256"] != stable_hash(job_contract(registration, job)):
        raise ValueError(f"Foreign checkpoint for {job['job_id']}")
    value = checkpoint["proposal_index"]
    if type(value) is not int or not 0 <= value <= job["search_config"]["proposal_budget"]:
        raise ValueError("Checkpoint outside registered budget")
    return value


def recovery_checkpoint(root, registration, job, *, on_retry=None):
    """After the worker exits, validate saved history before scheduling a retry."""
    folder = root / job["job_id"]
    with run_lock(folder / "run.lock"):
        checkpoint = h1_io.retry_access(
            lambda: load_checkpoint(folder, stable_hash(job_contract(registration, job))),
            on_retry=on_retry)
        h1_io.retry_access(lambda: load_state(folder, checkpoint["state_sha256"]),
                           on_retry=on_retry)
    return checkpoint


def child_command(root, job_id, session_proposals, session_seconds):
    return [sys.executable, "-u", "-m", "src.control.h1_ablation", "run",
            "--output-root", str(root), "--job-id", job_id,
            "--max-session-proposals", str(session_proposals),
            "--max-session-seconds", str(session_seconds)]


def run_matrix(root, *, workers=2, session_proposals=100, session_seconds=1800,
               poll_seconds=2, minimum_free_gib=10):
    if workers < 1 or session_proposals < 1 or session_seconds <= 0 or poll_seconds <= 0 or minimum_free_gib < 0:
        raise ValueError("Invalid operational limits")
    registration = validate_registration(root)
    with run_lock(root / "matrix_runner.lock"):
        return _run_locked(root, registration, workers, session_proposals, session_seconds,
                           poll_seconds, minimum_free_gib)


def _run_locked(root, registration, workers, session_proposals, session_seconds, poll_seconds, minimum_free_gib):
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    evidence = root / "runner_sessions" / run_id
    evidence.mkdir(parents=True)
    jobs = {j["job_id"]: j for j in registration["jobs"]}
    io_retries = []
    def retry_report(operation, path):
        def report(details):
            io_retries.append({"utc": now(), "operation": operation, "path": str(path), **details})
        return report
    positions = {name: progress(root, registration, job,
                               on_retry=retry_report("checkpoint_read", root / name / "checkpoint.json"))
                 for name, job in jobs.items()}
    pending = deque(name for name, job in jobs.items() if positions[name] < job["search_config"]["proposal_budget"])
    active = {}
    stopping = None
    recoveries = dict.fromkeys(jobs, 0)
    status = {"schema": "h1_matrix_runner_v1", "runner_id": run_id, "runner_pid": os.getpid(),
              "created_utc": now(), "registration_sha256": registration["sha256"],
              "runner_sha256_lf": text_hash(Path(__file__)), "purpose": registration["purpose"],
              "workers": workers, "session_proposals": session_proposals, "session_seconds": session_seconds,
              "minimum_free_gib": minimum_free_gib, "events": [], "runner_created_pause_files": [],
              "io_policy": "session_boundary_reads_bounded_retry_v1", "io_attempts": h1_io.IO_ATTEMPTS,
              "progress_policy": "last_exited_session_lower_bound_while_workers_active",
              "max_worker_recoveries_per_job": h1_io.WORKER_RECOVERIES,
              "worker_recoveries": recoveries, "io_retries": io_retries,
              "io_source_sha256_lf": text_hash(Path(h1_io.__file__)), "shutdown_errors": []}
    immutable_source = evidence / "run_h1_ablation_matrix.py"
    shutil.copyfile(Path(__file__), immutable_source)
    shutil.copyfile(h1_io.__file__, evidence / "h1_io.py")
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONUNBUFFERED="1")
    flags = (subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS) if os.name == "nt" else 0

    def publish(state):
        status.update(updated_utc=now(), status=state, reason=stopping,
                      progress=positions, pending=list(pending),
                      completed_jobs=sum(positions[n] == j["search_config"]["proposal_budget"] for n, j in jobs.items()),
                      total_jobs=len(jobs),
                      active={name: {"pid": entry["process"].pid, "log": str(entry["log"]),
                                     "starting_proposal": entry["start"]} for name, entry in active.items()})
        for path in (evidence / "status.json", root / "matrix_status.json"):
            h1_io.retry_access(lambda: atomic_json(path, status),
                               on_retry=retry_report("status_write", path))

    def refresh(name):
        value = progress(root, registration, jobs[name],
                         on_retry=retry_report("checkpoint_read", root / name / "checkpoint.json"))
        if value < positions[name]:
            raise ValueError(f"Checkpoint progress regressed: {name}")
        positions[name] = value

    def request_active_pause():
        for name in active:
            path = root / name / "PAUSE"
            path.parent.mkdir(parents=True, exist_ok=True)
            def create_pause():
                with path.open("x", encoding="utf-8") as stream:
                    stream.write(f"Matrix runner {run_id}: {stopping}\n")
            try:
                h1_io.retry_access(create_pause, on_retry=retry_report("pause_write", path))
                status["runner_created_pause_files"].append(str(path))
            except FileExistsError:
                pass

    try:
        publish("starting")
        while pending or active:
            if stopping is None:
                if (root / "PAUSE").exists():
                    stopping = "global_pause_file"
                elif any((root / name / "PAUSE").exists() for name in jobs):
                    stopping = "job_pause_file"
                elif shutil.disk_usage(root).free < minimum_free_gib * 2**30:
                    stopping = "disk_free_below_operational_floor"
                if stopping:
                    request_active_pause()
            while pending and len(active) < workers and stopping is None:
                name = pending.popleft()
                command = child_command(root, name, session_proposals, session_seconds)
                log = evidence / f"{len(status['events']):05d}_{name}.log"
                stream = log.open("x", encoding="utf-8")
                try:
                    process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream,
                                               stderr=subprocess.STDOUT, creationflags=flags)
                finally:
                    stream.close()
                active[name] = {"process": process, "log": log, "start": positions[name]}
                status["events"].append({"event": "started", "utc": now(), "job_id": name,
                                         "pid": process.pid, "command": command, "log": str(log),
                                         "starting_proposal": positions[name]})
                publish("running")
            for name, entry in list(active.items()):
                result = entry["process"].poll()
                if result is None:
                    continue
                refresh(name)
                del active[name]
                status["events"].append({"event": "exited", "utc": now(), "job_id": name,
                                         "exit_code": result, "proposal_index": positions[name]})
                if result:
                    if (stopping is None and recoveries[name] < h1_io.WORKER_RECOVERIES
                            and h1_io.checkpoint_access_failure(entry["log"], root / name / "checkpoint.json")):
                        checkpoint = recovery_checkpoint(root, registration, jobs[name],
                            on_retry=retry_report("recovery_audit", root / name))
                        recoveries[name] += 1
                        status["events"].append({"event": "checkpoint_access_recovery", "utc": now(),
                            "job_id": name, "attempt": recoveries[name], "failed_log": str(entry["log"]),
                            "checkpoint_sha256": checkpoint["sha256"],
                            "proposal_index": checkpoint["proposal_index"]})
                        pending.appendleft(name)
                        continue
                    stopping = f"child_failed:{name}:exit_{result}"
                    request_active_pause()
                elif positions[name] < jobs[name]["search_config"]["proposal_budget"]:
                    if positions[name] == entry["start"] and stopping is None:
                        stopping = f"child_made_no_progress:{name}"
                        request_active_pause()
                    pending.append(name)
            publish("stopping" if stopping else "running")
            if stopping and not active:
                state = "failed" if stopping.startswith("child_") else "paused"
                publish(state)
                return status
            if pending or active:
                time.sleep(poll_seconds)
        publish("search_complete")
        if registration["purpose"] == "production":
            output = evidence / "readout.json"
            command = [sys.executable, "-u", "-m", "src.validation.h1_ablation",
                       "--root", str(root), "--output", str(output)]
            status["readout_command"] = command
            publish("analyzing")
            with (evidence / "readout.log").open("x", encoding="utf-8") as stream:
                result = subprocess.run(command, cwd=ROOT, env=env, stdout=stream,
                                        stderr=subprocess.STDOUT, creationflags=flags)
            status["readout_exit_code"] = result.returncode
            status["readout_json"] = str(output)
            if result.returncode:
                stopping = "readout_failed"
                publish("failed")
                return status
        publish("complete")
        return status
    except BaseException as exc:
        stopping = f"runner_exception:{type(exc).__name__}:{exc}"
        def shutdown_error(operation, error):
            status["shutdown_errors"].append({"operation": operation, "error": repr(error)})
            print(f"Shutdown {operation}: {error!r}", file=sys.stderr, flush=True)
        try:
            request_active_pause()
        except Exception as error:
            shutdown_error("request_pause", error)
            # Only our own children; committed checkpoints survive termination.
            for entry in active.values():
                if entry["process"].poll() is None:
                    entry["process"].terminate()
        try:
            publish("stopping")
        except Exception as error:
            shutdown_error("publish_stopping", error)
        for name, entry in active.items():
            result = entry["process"].wait()
            status["events"].append({"event": "drained", "utc": now(), "job_id": name, "exit_code": result})
            try:
                refresh(name)
            except Exception as error:
                shutdown_error(f"final_progress:{name}", error)
        active.clear()
        try:
            publish("failed")
        except Exception as error:
            shutdown_error("publish_failed", error)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--session-proposals", type=int, default=100)
    parser.add_argument("--session-seconds", type=float, default=1800)
    args = parser.parse_args()
    result = run_matrix(args.output_root.resolve(), workers=args.workers,
                        session_proposals=args.session_proposals, session_seconds=args.session_seconds)
    print(json.dumps({k: result[k] for k in ("status", "reason", "completed_jobs", "total_jobs")}), flush=True)
    if result["status"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
