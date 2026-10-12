"""Descriptive equal-proposal-budget readout for Decision 047 only."""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from src.retrodiction.h1_checkpointed import SearchConfig, audit_search, atomic_json


def panel_statistics(trajectories, protocol):
    """One record per source/seed/arm; never treat overlapping pairs as samples."""
    languages = protocol["romance"] + protocol["controls"]
    romance = set(protocol["romance"])
    seeds = protocol["search_seeds"]
    points = protocol["proposal_checkpoints"]
    grid = np.array(points, dtype=float) / protocol["proposal_budget"]
    if not np.isfinite(grid).all() or grid[0] != 0 or grid[-1] != 1 or np.any(np.diff(grid) <= 0):
        raise ValueError("Invalid proposal grid")
    per_seed = []
    pair_records = []
    for seed in seeds:
        arm_values = {}
        for arm in protocol["form_reward_modes"]:
            values = {}
            for language in languages:
                key = (language, seed, arm)
                if key not in trajectories:
                    raise ValueError(f"Missing main trajectory: {key}")
                values[language] = np.asarray(trajectories[key], dtype=float)
                if values[language].shape != (len(points), 6) or not np.isfinite(values[language]).all():
                    raise ValueError(f"Invalid trajectory features: {key}")
            within, other = [], []
            for a, b in itertools.combinations(languages, 2):
                distances = np.linalg.norm(values[a] - values[b], axis=1)
                if distances[0] <= 0:
                    raise ValueError(f"Zero initial pair distance: {a}, {b}")
                normalized = distances / distances[0]
                (within if a in romance and b in romance else other).append(normalized)
                pair_records.append({"seed": seed, "arm": arm, "languages": [a, b],
                                     "romance_pair": a in romance and b in romance,
                                     "normalized_distances": normalized.tolist(),
                                     "proposal_budget_auc": float(np.trapezoid(normalized, grid))})
            effect = np.mean(other, axis=0) - np.mean(within, axis=0)
            arm_values[arm] = {"checkpoint_contrasts": effect.tolist(),
                               "proposal_budget_auc_contrast": float(np.trapezoid(effect, grid))}
        difference = (np.array(arm_values["full"]["checkpoint_contrasts"])
                      - np.array(arm_values["disabled"]["checkpoint_contrasts"]))
        per_seed.append({"search_seed": seed, "arms": arm_values,
                         "full_minus_disabled_checkpoints": difference.tolist(),
                         "full_minus_disabled_auc": float(np.trapezoid(difference, grid))})
    return {"proposal_checkpoints": points, "per_seed": per_seed,
            "mean_full_minus_disabled_checkpoints": np.mean(
                [r["full_minus_disabled_checkpoints"] for r in per_seed], axis=0).tolist(),
            "mean_full_minus_disabled_auc": float(np.mean([r["full_minus_disabled_auc"] for r in per_seed])),
            "pair_records": pair_records}


def analyze(root):
    from src.control.h1_ablation import validate_registration, make_engine, job_contract
    registration = validate_registration(root)
    if registration["purpose"] != "production":
        raise ValueError("Pilot registrations are excluded from scientific readout")
    protocol = registration["identity"]["protocol"]
    main = set(protocol["romance"] + protocol["controls"])
    trajectories, diagnostics, stress, unavailable = {}, [], [], []
    for job in registration["jobs"]:
        config = SearchConfig.from_dict(job["search_config"])
        output = root / job["job_id"]
        is_main = config.language in main
        try:
            if not (output / "checkpoint.json").exists():
                raise ValueError("Missing checkpoint")
            # Readers hold the writer lock so a concurrent commit cannot produce
            # a mixture of checkpoint generations during the independent replay.
            from src.retrodiction.h1_run_lock import run_lock
            with threadpool_limits(limits=1), run_lock(output / "run.lock"):
                engine = make_engine(config, output)
                audit = audit_search(engine, config, output, job_contract(registration, job))
            if not audit["complete"]:
                raise ValueError("Run paused/incomplete; budget not reached")
            record = {"job_id": job["job_id"], "search_config": config.to_dict(), "audit": audit}
            if is_main:
                trajectories[(config.language, config.search_seed, config.form_reward_mode)] = [
                    audit["checkpoints"][str(p)]["measurement"]["z_6d"] for p in config.proposal_checkpoints]
                diagnostics.append(record)
            else:
                stress.append(record)
        except (ValueError, KeyError, FileNotFoundError) as exc:
            if is_main:
                raise ValueError(f"Main run not evaluable: {job['job_id']}: {exc}") from exc
            unavailable.append({"job_id": job["job_id"], "reason": str(exc)})
    return {"schema": "h1_form_reward_ablation_readout_v1", "status": "descriptive_complete",
            "registration_sha256": registration["sha256"], "question": protocol["question"],
            "claim_scope": protocol["scope"], "inference": protocol["inference"],
            "statistics": panel_statistics(trajectories, protocol), "main_diagnostics": diagnostics,
            "stress_diagnostics": stress, "unavailable_stress_runs": unavailable}


def render_report(result):
    if result["status"] == "not_evaluable":
        return "# Form-reward ablation: not evaluable\n\n" + result["reason"] + "\n\nNo scientific verdict.\n"
    statistics = result["statistics"]
    lines = ["# Fixed-budget form-reward ablation", "", result["question"], "",
             "Descriptive comparison of this corpus panel. No unconditional significance or ancestry verdict.", "",
             "Positive full-minus-disabled means a larger Romance-versus-other convergence contrast under full rewards.", "",
             "| Search seed | Full proposal AUC contrast | Disabled contrast | Full minus disabled |",
             "|---|---:|---:|---:|"]
    for row in statistics["per_seed"]:
        lines.append(f"| {row['search_seed']} | {row['arms']['full']['proposal_budget_auc_contrast']:.8g} | "
                     f"{row['arms']['disabled']['proposal_budget_auc_contrast']:.8g} | {row['full_minus_disabled_auc']:.8g} |")
    lines.extend(["", f"Mean arm difference: {statistics['mean_full_minus_disabled_auc']:.8g}.", "",
                  "The JSON contains every checkpoint contrast, raw feature/diversity diagnostic, and candidate counter.",
                  "Latin-informed mutation guidance remains in both arms. This is not a target-free comparison.",
                  f"Unavailable stress runs: {len(result['unavailable_stress_runs'])}. They do not alter main-panel denominators.", ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.suffix != ".json" or args.output.exists() or args.output.with_suffix(".md").exists():
        parser.error("Use a fresh .json output path and matching .md path")
    try:
        result = analyze(args.root.resolve())
    except (ValueError, KeyError, FileNotFoundError) as exc:
        result = {"status": "not_evaluable", "reason": str(exc)}
    atomic_json(args.output, result)
    args.output.with_suffix(".md").write_text(render_report(result), encoding="utf-8")
    if result["status"] == "not_evaluable":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
