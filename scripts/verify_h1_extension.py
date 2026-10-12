"""Real-corpus, finite validation of Decision 047; never production evidence.

Run with: python scripts/verify_h1_extension.py --output-root <new-directory>
"""
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

from threadpoolctl import threadpool_limits
from src.control.h1_ablation import register, run_job, make_engine, job_contract, make_plan
from src.retrodiction.h1_checkpointed import SearchConfig, atomic_json, audit_search, read_json
from src.retrodiction.h1_objective import text_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    if os.name == "nt":
        import ctypes
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
    results = {"purpose": "finite_engineering_validation_excluded_from_scientific_readout",
               "script_sha256_lf": text_hash(Path(__file__)), "runs": [], "comparisons": []}
    registration = register(root / "paused", "pilot")
    reference = register(root / "uninterrupted", "pilot")
    atomic_json(root / "production_plan.json", make_plan(ROOT / "data/h1_ablation_production_20261001"))
    for job in registration["jobs"]:
        job_id = job["job_id"]
        started = time.perf_counter()
        first = run_job(root / "paused", job_id, max_session_proposals=1)
        assert first["status"] == "paused" and first["proposal_index"] == 1
        resumed = run_job(root / "paused", job_id)
        assert resumed["status"] == "complete" and resumed["proposal_index"] == 4
        cfg = SearchConfig.from_dict(job["search_config"])
        output = root / "paused" / job_id
        with threadpool_limits(limits=1):
            audit = audit_search(make_engine(cfg, output), cfg, output, job_contract(registration, job))
        atomic_json(output / "verification.json", audit)
        results["runs"].append({"job_id": job_id, "first": first, "resumed": resumed,
                                "audit": audit, "wall_seconds": time.perf_counter() - started})
        atomic_json(root / "verification_results.json", results)
        print(json.dumps({"job_id": job_id, "states_replayed": audit["states_replayed"],
                          "wall_seconds": results["runs"][-1]["wall_seconds"]}), flush=True)
        if cfg.language == "french":
            run_job(root / "uninterrupted", job_id)
            expected = root / "uninterrupted" / job_id
            a, b = read_json(output / "checkpoint.json"), read_json(expected / "checkpoint.json")
            fields = ("proposal_index", "accepted_states", "stagnation", "state_sha256", "rng_state", "counters")
            assert all(a[key] == b[key] for key in fields)
            for index in range(5):
                a = read_json(output / "proposals" / f"{index:09d}.json")
                b = read_json(expected / "proposals" / f"{index:09d}.json")
                assert all(a[key] == b[key] for key in ("proposal_index", "before_state", "after_state", "attempts", "counters"))
            assert {p.name: p.read_bytes() for p in (output / "states").glob("*.json")} == {
                p.name: p.read_bytes() for p in (expected / "states").glob("*.json")}
            results["comparisons"].append({"job_id": job_id, "identical_proposals_states_rng_and_counters": True})
            atomic_json(root / "verification_results.json", results)
    cli = subprocess.run([sys.executable, "-m", "src.validation.h1_ablation", "--root", str(root / "paused"),
                          "--output", str(root / "pilot_readout_exclusion.json")], cwd=ROOT, capture_output=True,
                         text=True, timeout=60, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    assert cli.returncode == 2
    assert read_json(root / "pilot_readout_exclusion.json")["status"] == "not_evaluable"
    results["pilot_readout_exit_code"] = cli.returncode
    results["complete"] = True
    results["production_launched"] = False
    atomic_json(root / "verification_results.json", results)
    print("All finite real-corpus checks passed; no production launched.", flush=True)


if __name__ == "__main__":
    main()
