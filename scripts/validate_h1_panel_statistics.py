"""Reproducible synthetic validation of Decision 046 (no real-run outcomes)."""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.validation.h1_convergence import ROMANCE, CONTROLS, SEEDS, whole_language_reference, panel_summary
from src.validation.h1_protocol import protocol_identity
from src.retrodiction.h1_objective import text_hash


def simulation(repetitions=1000, seed=20261001):
    if repetitions <= 0:
        raise ValueError("Repetitions must be positive")
    rng = np.random.default_rng(seed)
    languages = ROMANCE + CONTROLS
    edges = list(itertools.combinations(languages, 2))
    scenarios = {}
    for scenario in ("exchangeable_null", "within_romance_alternative", "group_correlated_nuisance"):
        records = []
        exact_orbit_rates = []
        for trial in range(repetitions):
            # A shared language effect creates dependent edges and repeated seeds.
            language_effect = dict(zip(languages, rng.normal(0, 0.2, len(languages))))
            if scenario == "group_correlated_nuisance":
                # No genealogical mechanism: a corpus nuisance correlates with labels.
                for language in ROMANCE:
                    language_effect[language] -= 0.65
            rows = []
            for run_seed in SEEDS:
                seed_language = dict(zip(languages, rng.normal(0, 0.03, len(languages))))
                for a, b in edges:
                    log_auc = language_effect[a] + language_effect[b] + seed_language[a] + seed_language[b]
                    if scenario == "within_romance_alternative" and a in ROMANCE and b in ROMANCE:
                        log_auc -= 0.9
                    rows.append({"seed": run_seed, "left": a, "right": b, "auc": float(np.exp(log_auc))})
            reference = whole_language_reference(rows)
            primary = panel_summary(rows)
            records.append({"trial": trial, "effect": primary["effect"],
                            "tail_fraction": reference["upper_tail_fraction"],
                            "direction_all_seeds": primary["direction_replicates_all_seeds"]})
            if scenario == "exchangeable_null":
                stats = np.array([a["effect"] for a in reference["allocations"]])
                tails = np.array([np.mean(stats >= t - 1e-12 * max(1, abs(t))) for t in stats])
                rate = float(np.mean(tails < 0.05))
                if rate > 6 / 126 + 1e-15:
                    raise AssertionError("Exact uniform-label calibration failed")
                exact_orbit_rates.append(rate)
        rejected = sum(r["tail_fraction"] < 0.05 for r in records)
        scenarios[scenario] = {
            "trials": repetitions, "tail_below_0_05_count": rejected,
            "tail_below_0_05_fraction": rejected / repetitions,
            "consistent_direction_count": sum(r["direction_all_seeds"] for r in records),
            "exact_orbit_rejection_rate_max": max(exact_orbit_rates) if exact_orbit_rates else None,
            "records": records,
        }
    return {"schema": "h1_panel_synthetic_validation_v1", "seed": seed,
            "protocol_identity": protocol_identity(), "script_sha256_lf": text_hash(Path(__file__)),
            "scenarios": scenarios,
            "interpretation": "Checks conditional label-reference calculation; does not establish real-language exchangeability, genealogical specificity or empirical power",
            "generators": {"null": "iid language effects shared by all incident edges and all three seeds",
                           "alternative": "additional -0.9 log-AUC for within-Romance pairs",
                           "nuisance": "-0.65 corpus effect for each Romance node; label exchangeability deliberately violated"}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261001)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Preserve earlier validation output; choose a new file")
    result = simulation(args.repetitions, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({k: {a: b for a, b in v.items() if a != "records"}
                      for k, v in result["scenarios"].items()}, indent=2))
