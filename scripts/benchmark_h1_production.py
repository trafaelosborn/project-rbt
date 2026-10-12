"""Bounded production-setting benchmark; never produces an E1 verdict."""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.control.h1_matrix import build_jobs
from src.retrodiction.h1_objective import PROJECT_ROOT, text_hash


def run(root: Path, proposals: int = 1000, timeout_seconds: int = 1800, profile=False):
    if proposals <= 0 or timeout_seconds <= 0:
        raise ValueError("Benchmark limits must be positive")
    if root.exists():
        raise ValueError("Use a fresh evidence directory")
    jobs = [replace(j, total_target_proposals=proposals)
            for j in build_jobs(root)
            if j.language in ("french", "finnish") and j.seed == 42]
    root.mkdir(parents=True)
    commands = {j.job_id: ([sys.executable, "-m", "cProfile", "-o", str(root / f"{j.job_id}.prof"), *j.command()[1:]]
                           if profile else j.command()) for j in jobs}
    plan = {"purpose": "bounded_throughput_only_not_scientific_evidence",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "script_sha256_lf": text_hash(Path(__file__)),
            "proposals": proposals, "timeout_seconds_per_job": timeout_seconds,
            "profiled": profile, "jobs": [{**j.to_dict(), "command": commands[j.job_id]} for j in jobs]}
    (root / "benchmark_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    env = dict(os.environ)
    env.update({k: "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")})
    results = []
    for job in jobs:
        log = root / f"{job.job_id}.log"
        print(f"Benchmark starting: {job.job_id}, {proposals} proposals, 16 candidates", flush=True)
        started = time.perf_counter()
        status, code = "completed", None
        with log.open("w", encoding="utf-8") as output:
            try:
                completed = subprocess.run(commands[job.job_id], cwd=PROJECT_ROOT, env=env,
                                           stdout=output, stderr=subprocess.STDOUT,
                                           timeout=timeout_seconds,
                                           creationflags=(subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS)
                                           if os.name == "nt" else 0)
                code = completed.returncode
                if code:
                    status = "failed"
            except subprocess.TimeoutExpired:
                status = "timeout"
        seconds = time.perf_counter() - started
        result = {"job_id": job.job_id, "status": status, "exit_code": code,
                  "wall_seconds_including_startup_audit": seconds, "log": str(log),
                  "manifest": str(job.manifest_path)}
        if status == "completed":
            manifest = json.loads(job.manifest_path.read_text(encoding="utf-8"))
            result["manifest_status"] = manifest["status"]
            result["blocks"] = len(manifest["blocks"])
            result["manifest_sha256_lf"] = text_hash(job.manifest_path)
            actual = manifest["completed_block_proposals"]
            result["actual_proposals"] = actual
            result["proposals_per_second"] = actual / seconds
            result["seconds_per_proposal"] = seconds / actual if actual else None
        results.append(result)
        (root / "benchmark_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(json.dumps(result), flush=True)
        if status != "completed":
            raise RuntimeError(f"Benchmark {status}: {log}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--proposals", type=int, default=1000)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    run(args.output_root.resolve(), args.proposals, args.timeout_seconds, args.profile)
