"""Real Windows checkpoint contention, queue recovery, and trajectory equivalence.

Creates only fresh pilot roots. Never point this at a production directory.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_h1_ablation_matrix import run_matrix
from src.control.h1_ablation import register
from src.retrodiction.h1_checkpointed import atomic_json, read_json
from src.retrodiction.h1_objective import text_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if os.name != "nt":
        parser.error("This verification requires actual Windows sharing semantics")
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    split, reference = root / "contended", root / "reference"
    register(split, "pilot")
    register(reference, "pilot")
    receipt = {"purpose": "finite_pilot_io_recovery_validation", "complete": False,
               "script_sha256_lf": text_hash(Path(__file__)), "calls": []}
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    flags = subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS
    def call(destination, job, label, cap=None):
        command = [sys.executable, "-u", "-m", "src.control.h1_ablation", "run",
                   "--output-root", str(destination), "--job-id", job]
        if cap is not None:
            command += ["--max-session-proposals", str(cap)]
        with (root / f"{label}.log").open("x", encoding="utf-8") as stream:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                    creationflags=flags, timeout=180)
        receipt["calls"].append({"command": command, "exit_code": result.returncode, "log": f"{label}.log"})
        atomic_json(root / "verification_results.json", receipt)
        if result.returncode:
            raise RuntimeError(f"Pilot child failed: {label}")
    job = "french__corpus_42__search_42__full"
    call(split, job, "seed_checkpoint", cap=1)
    reader = (split / job / "checkpoint.json").open(encoding="utf-8")
    released, stop = threading.Event(), threading.Event()
    contention = {}
    def release_after_observed_failure():
        deadline = time.monotonic() + 180
        try:
            while not stop.wait(.05) and time.monotonic() < deadline:
                for log in (split / "runner_sessions").glob(f"*/*{job}.log"):
                    try:
                        content = log.read_text(encoding="utf-8")
                    except OSError:
                        continue
                    if "PermissionError:" in content:
                        contention["failed_log"] = str(log)
                        contention["checkpoint_at_failure"] = json.load(reader)["proposal_index"]
                        released.set()
                        return
        finally:
            reader.close()
    watcher = threading.Thread(target=release_after_observed_failure, daemon=True)
    watcher.start()
    try:
        status = run_matrix(split, workers=2, session_proposals=2, session_seconds=60, poll_seconds=.1)
    finally:
        stop.set()
        watcher.join(timeout=5)
    assert not watcher.is_alive()
    assert released.is_set(), "Did not reproduce the intended real Windows write failure"
    assert status["status"] == "complete" and status["completed_jobs"] == 4
    assert status["worker_recoveries"][job] == 1
    receipt["contention"] = contention
    receipt["queue_status"] = status
    receipt["comparisons"] = []
    for arm in ("full", "disabled"):
        name = f"french__corpus_42__search_42__{arm}"
        call(reference, name, f"reference_{arm}")
        a, b = split / name, reference / name
        ca, cb = read_json(a / "checkpoint.json"), read_json(b / "checkpoint.json")
        for key in ("proposal_index", "accepted_states", "stagnation", "state_sha256", "rng_state", "counters"):
            assert ca[key] == cb[key], (arm, key)
        for index in range(5):
            ea = read_json(a / "proposals" / f"{index:09d}.json")
            eb = read_json(b / "proposals" / f"{index:09d}.json")
            for key in ("proposal_index", "before_state", "after_state", "attempts", "counters"):
                assert ea[key] == eb[key], (arm, index, key)
        assert {p.name: p.read_bytes() for p in (a / "states").glob("*.json")} == {
            p.name: p.read_bytes() for p in (b / "states").glob("*.json")}
        receipt["comparisons"].append({"arm": arm, "scientific_trajectory_identical": True,
                                        "contract_hashes_differ_for_distinct_output_roots": True})
        atomic_json(root / "verification_results.json", receipt)
    receipt["complete"] = True
    atomic_json(root / "verification_results.json", receipt)
    print(json.dumps({"complete": True, "jobs_completed": 4, "automatic_recoveries": 1,
                      "comparisons": receipt["comparisons"]}), flush=True)


if __name__ == "__main__":
    main()
