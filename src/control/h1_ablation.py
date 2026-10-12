"""Register, inspect, and run the Decision 047 checkpointed ablation extension."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import platform
import sys

import numpy as np
from threadpoolctl import threadpool_limits

from src.retrodiction.h1_checkpointed import (
    SearchConfig, atomic_json, engine_config, read_json, run_search, seal, verify,
)
from src.retrodiction.h1_objective import (
    PROJECT_ROOT, H1Objective, FREEZE_COMMIT, FEATURE_SHA256, stable_hash, text_hash,
)
from src.validation.h1_protocol import reference_input_hashes

PROTOCOL = "docs/registrations/h1_form_reward_ablation_v1.json"
DECISION = "docs/decisions/047_h1_continuous_search_and_form_reward_ablation.md"
PINNED = {
    PROTOCOL: "e6ec3f43f261c48252cca7c2ebc55ac8ed2484e1a558ddea84024f94fe8ff9a9",
    DECISION: "006470dd93beab9e30be367297ae7fa8d993bf257c51b3d3eb81b9942a04f5e4",
}
FILES = ("src/retrodiction/h1_checkpointed.py", "src/retrodiction/h1_run_lock.py",
         "src/control/h1_ablation.py", "src/validation/h1_ablation.py",
         "src/validation/h1_protocol.py", "src/validation/h1_convergence.py")
SIDECAR = "ablation_registration.json"


def identity():
    for relative, digest in PINNED.items():
        if text_hash(PROJECT_ROOT / relative) != digest:
            raise ValueError(f"Pinned ablation protocol changed: {relative}")
    obj = H1Objective.load(FREEZE_COMMIT, FEATURE_SHA256, 20000)
    return {"protocol": read_json(PROJECT_ROOT / PROTOCOL), "protocol_hashes_lf": PINNED,
            "implementation_hashes_lf": {p: text_hash(PROJECT_ROOT / p) for p in FILES},
            "objective": obj.provenance, "reference_inputs_sha256_lf": reference_input_hashes(),
            "runtime": {"python": sys.version, "numpy": np.__version__, "scipy": version("scipy"),
                        "threadpoolctl": version("threadpoolctl"), "numerical_library_threads": 1,
                        "platform": platform.platform(), "machine": platform.machine(),
                        "rng": "numpy.random.PCG64"}}


def source_path(language):
    bank = read_json(PROJECT_ROOT / "data/v2/corpora_manifest.json")
    matches = [r for r in bank["corpora"] if r["language"] == language and r["status"] == "ready"]
    if len(matches) != 1:
        raise ValueError(f"No unique frozen source for {language}")
    return (PROJECT_ROOT / matches[0]["tokens_path"]).resolve()


def sampled_source(config):
    from src.validation.static_feature_matrix import subsample_sequences
    return subsample_sequences(read_json(source_path(config.language))["sequences"],
                               token_budget=20000, seed=config.corpus_seed, corpus_id=config.language)


def jobs(root, purpose):
    if purpose not in {"production", "pilot"}:
        raise ValueError("Unknown registration purpose")
    protocol = read_json(PROJECT_ROOT / PROTOCOL)
    languages = (["french", "finnish"] if purpose == "pilot" else
                 protocol["romance"] + protocol["controls"] + protocol["stress_sources"])
    result = []
    for language in languages:
        sample = None
        source = source_path(language)
        source_hash = text_hash(source)
        for seed in ([42] if purpose == "pilot" else protocol["search_seeds"]):
            for arm in protocol["form_reward_modes"]:
                cfg = SearchConfig(language, protocol["corpus_seed"], seed, arm,
                                   4 if purpose == "pilot" else protocol["proposal_budget"],
                                   2 if purpose == "pilot" else protocol["n_candidates"],
                                   (0, 2, 4) if purpose == "pilot" else tuple(protocol["proposal_checkpoints"]))
                job_id = f"{language}__corpus_{cfg.corpus_seed}__search_{seed}__{arm}"
                if sample is None:
                    sample = sampled_source(cfg)
                effective = engine_config(cfg, source, root / job_id)
                result.append({"job_id": job_id, "search_config": cfg.to_dict(),
                               "engine_config": effective.to_dict(), "source_path": str(source),
                               "source_sha256_lf": source_hash,
                               "sampled_corpus_sha256": stable_hash(sample)})
    return result


def make_plan(root, purpose="production"):
    return {"schema": "h1_ablation_registration_v1", "purpose": purpose,
            "root": str(root.resolve()), "identity": identity(), "jobs": jobs(root.resolve(), purpose)}


def register(root, purpose="production"):
    root = root.resolve()
    plan = make_plan(root, purpose)
    plan["created_utc"] = datetime.now(timezone.utc).isoformat()
    payload = seal(plan)
    root.mkdir(parents=True, exist_ok=False)
    atomic_json(root / SIDECAR, payload)
    return payload


def validate_registration(root):
    payload = verify(read_json(root / SIDECAR))
    expected = make_plan(root, payload["purpose"])
    if any(payload.get(key) != value for key, value in expected.items()):
        raise ValueError("Registered code, inputs, runtime, job settings, or protocol changed")
    return payload


def job_contract(registration, job):
    return {**job, "registration_sha256": registration["sha256"],
            "identity": registration["identity"], "purpose": registration["purpose"]}


def make_engine(config, output):
    from src.retrodiction.engine_reinforced import LatinReference
    from src.retrodiction.engine_reinforced_v2 import LatinFormReference
    from src.retrodiction.engine_reinforced_v5 import RelationalReinforcedRetrodictionEngineV5
    from src.retrodiction.similarity import ReferenceSet
    cfg = engine_config(config, source_path(config.language), output)
    return RelationalReinforcedRetrodictionEngineV5(
        config.language, [], LatinReference(), LatinFormReference(), cfg, output, ReferenceSet())


def run_job(root, job_id, *, max_session_proposals=None, max_session_seconds=None):
    registration = validate_registration(root)
    selected = [j for j in registration["jobs"] if j["job_id"] == job_id]
    if len(selected) != 1:
        raise ValueError("Job is absent from the pre-launch registration")
    job = selected[0]
    config = SearchConfig.from_dict(job["search_config"])
    sample = sampled_source(config)
    if stable_hash(sample) != job["sampled_corpus_sha256"]:
        raise ValueError("Initial corpus differs from its pre-launch identity")
    output = root / job_id
    with threadpool_limits(limits=1):
        engine = make_engine(config, output)
        return run_search(engine, config, output, job_contract(registration, job), sample,
                          max_session_proposals=max_session_proposals,
                          max_session_seconds=max_session_seconds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["plan", "register", "run", "status"])
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--purpose", choices=["production", "pilot"], default="production")
    parser.add_argument("--plan-file", type=Path)
    parser.add_argument("--job-id")
    parser.add_argument("--max-session-proposals", type=int)
    parser.add_argument("--max-session-seconds", type=float)
    args = parser.parse_args()
    root = args.output_root.resolve()
    if args.mode == "plan":
        if args.plan_file is None or args.plan_file.exists():
            parser.error("plan requires a new --plan-file")
        atomic_json(args.plan_file, make_plan(root, args.purpose))
        print("Plan written; no runs launched.")
    elif args.mode == "register":
        registered = register(root, args.purpose)
        print(f"Registered {len(registered['jobs'])} jobs; no runs launched.")
    elif args.mode == "run":
        if not args.job_id:
            parser.error("run requires --job-id")
        print(json.dumps(run_job(root, args.job_id, max_session_proposals=args.max_session_proposals,
                                 max_session_seconds=args.max_session_seconds), indent=2))
    else:
        registered = validate_registration(root)
        for job in registered["jobs"]:
            path = root / job["job_id"] / "checkpoint.json"
            point = verify(read_json(path))["proposal_index"] if path.exists() else 0
            print(f"{job['job_id']}: {point}/{job['search_config']['proposal_budget']}")


if __name__ == "__main__":
    main()
