"""Write a corrected H1 run plan or execute a bounded production-launcher pilot.

This command never starts an unbounded production matrix automatically.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

from src.control.multi_run_supervisor import SupervisorJob
from src.retrodiction.h1_objective import H1Objective, FREEZE_COMMIT, FEATURE_SHA256, PROJECT_ROOT
from src.validation.h1_convergence import ROMANCE, CONTROLS, SEEDS
from src.validation.h1_protocol import protocol_identity, write_registration


def build_jobs(root: Path, *, pilot=False):
    H1Objective.load(FREEZE_COMMIT, FEATURE_SHA256, 20_000)
    bank = json.loads((PROJECT_ROOT / "data/v2/corpora_manifest.json").read_text(encoding="utf-8"))
    languages = ("french", "finnish") if pilot else ROMANCE + CONTROLS + ("markov_null", "klingon")
    jobs = []
    for language in languages:
        row = next(r for r in bank["corpora"] if r["language"] == language and r["status"] == "ready")
        for seed in ((42,) if pilot else SEEDS):
            jobs.append(SupervisorJob(
                language=language, seed=seed, start_corpus=(PROJECT_ROOT / row["tokens_path"]).resolve(),
                output_dir=root.resolve() / language / f"seed_{seed}",
                total_target_proposals=4 if pilot else 0,
                block_proposals=2 if pilot else 1000, candidate_count=2 if pilot else 16,
                features_freeze_commit=FREEZE_COMMIT, features_freeze_sha256=FEATURE_SHA256,
                corpus_budget=20_000, live_event_mode="all" if pilot else "accepted_only",
                validator_snapshot_every_blocks=1 if pilot else 0,
            ))
    return jobs


def write_plan(root: Path, plan_file: Path, *, pilot=False):
    jobs = build_jobs(root, pilot=pilot)
    payload = {
        "schema": "corrected_h1_launch_plan_v2", "purpose": "pilot" if pilot else "production_plan_only",
        "analysis_registration": protocol_identity(),
        "objective": H1Objective.load(FREEZE_COMMIT, FEATURE_SHA256, 20000).provenance,
        "job_count": len(jobs),
        "primary_trajectories": 27 if not pilot else 0,
        "claim_scope": "Decision 046 fixed-panel E1 comparison; no unconditional H1 significance verdict",
        "remaining_scientific_work": "Random-init robustness, other budgets and E2-E4 are explicitly deferred; full original H1 is not completed.",
        "jobs": [{**j.to_dict(), "command": j.command()} for j in jobs],
    }
    plan_file.parent.mkdir(parents=True, exist_ok=True)
    if plan_file.exists():
        raise ValueError(f"Refusing to replace existing launch plan: {plan_file}")
    plan_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return jobs


def run_pilot(root: Path):
    if root.exists():
        raise ValueError("Pilot output root must be new; preserve earlier trials")
    jobs = write_plan(root, root / "launch_plan.json", pilot=True)
    env = dict(os.environ)
    env.update({k: "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")})
    results = []
    for job in jobs:
        log_path = root / f"{job.job_id}.log"
        started = time.perf_counter()
        with log_path.open("w", encoding="utf-8") as output:
            completed = subprocess.run(job.command(), cwd=PROJECT_ROOT, env=env,
                                       stdout=output, stderr=subprocess.STDOUT, timeout=600,
                                       creationflags=(subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS) if os.name == "nt" else 0)
        result = {"job_id": job.job_id, "exit_code": completed.returncode,
                  "wall_seconds": time.perf_counter() - started, "log": str(log_path)}
        results.append(result)
        (root / "pilot_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(json.dumps(result), flush=True)
        if completed.returncode:
            raise RuntimeError(f"Pilot failed; inspect {log_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["plan", "pilot", "register"])
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--plan-file", type=Path)
    args = parser.parse_args()
    if args.mode == "pilot":
        run_pilot(args.output_root.resolve())
    elif args.mode == "register":
        if args.plan_file is None:
            parser.error("register requires --plan-file")
        if args.plan_file.exists():
            raise ValueError("Use a new launch plan path")
        jobs = build_jobs(args.output_root)
        registration = write_registration(args.output_root, jobs)
        write_plan(args.output_root, args.plan_file)
        print(f"Registered {len(jobs)} jobs: {registration['registration_sha256']}; no production runs launched.")
    else:
        if args.plan_file is None:
            parser.error("plan requires --plan-file")
        jobs = write_plan(args.output_root, args.plan_file)
        print(f"Wrote {len(jobs)} jobs; no production runs launched.")


if __name__ == "__main__":
    main()
