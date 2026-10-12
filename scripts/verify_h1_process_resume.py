"""Verify actual process-exit/resume equivalence with different Python hash seeds."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.control.h1_ablation import register
from src.retrodiction.h1_checkpointed import atomic_json, read_json
from src.retrodiction.h1_objective import text_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    register(root / "split", "pilot")
    register(root / "whole", "pilot")
    receipt = {"purpose": "finite_process_resume_validation_only", "script_sha256_lf": text_hash(Path(__file__)),
               "calls": [], "comparisons": [], "production_launched": False}
    for arm in ("full", "disabled"):
        job = f"french__corpus_42__search_42__{arm}"
        for index, (destination, cap, hash_seed) in enumerate((("split", 1, "101"), ("split", None, "202"), ("whole", None, "303"))):
            command = [sys.executable, "-m", "src.control.h1_ablation", "run", "--output-root", str(root / destination), "--job-id", job]
            if cap is not None:
                command.extend(["--max-session-proposals", str(cap)])
            env = dict(os.environ, PYTHONHASHSEED=hash_seed, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
            log = root / f"{arm}_{index}.log"
            started = time.perf_counter()
            with log.open("w", encoding="utf-8") as stream:
                completed = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                           timeout=180, creationflags=(subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS) if os.name == "nt" else 0)
            receipt["calls"].append({"command": command, "python_hash_seed": hash_seed,
                                     "wall_seconds": time.perf_counter() - started, "exit_code": completed.returncode,
                                     "log": str(log)})
            atomic_json(root / "verification_results.json", receipt)
            if completed.returncode:
                raise RuntimeError(f"Process failed: {log}")
        split, whole = root / "split" / job, root / "whole" / job
        a, b = read_json(split / "checkpoint.json"), read_json(whole / "checkpoint.json")
        fields = ("proposal_index", "accepted_states", "stagnation", "state_sha256", "rng_state", "counters")
        assert all(a[key] == b[key] for key in fields)
        for index in range(5):
            a = read_json(split / "proposals" / f"{index:09d}.json")
            b = read_json(whole / "proposals" / f"{index:09d}.json")
            assert all(a[key] == b[key] for key in ("proposal_index", "before_state", "after_state", "attempts", "counters"))
        assert {p.name: p.read_bytes() for p in (split / "states").glob("*.json")} == {
            p.name: p.read_bytes() for p in (whole / "states").glob("*.json")}
        receipt["comparisons"].append({"arm": arm, "identical_across_process_exit_and_hash_seeds": True})
        atomic_json(root / "verification_results.json", receipt)
        print(f"{arm}: exact process-resume equivalence passed", flush=True)
    receipt["complete"] = True
    atomic_json(root / "verification_results.json", receipt)


if __name__ == "__main__":
    main()
