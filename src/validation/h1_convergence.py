"""Amended E1 fixed-panel readout for verified, completed corrected trajectories.

Decision 046 replaces the dependent-pair significance gate with a panel effect
and an explicitly conditional whole-language label reference.
Bounded pilots and legacy matrices cannot produce a hypothesis verdict here.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np

from src.validation.h1_protocol import validate_registration, validate_registered_manifest

ROMANCE = ("french", "spanish", "italian", "portuguese", "romanian")
CONTROLS = ("finnish", "turkish", "vietnamese", "sumerian")
SEEDS = (42, 1729, 8675309)


def exact_mann_whitney_less(left, right):
    """Historical calculation only: NOT valid inference for shared E1 pairs."""
    x, y = list(map(float, left)), list(map(float, right))
    if not x or not y or not all(math.isfinite(v) for v in x + y):
        raise ValueError("Mann-Whitney requires finite, nonempty groups")
    values = x + y
    # Twice the average rank is integral, even for ties.
    ranks = [2 * sum(w < v for w in values) + sum(w == v for w in values) + 1 for v in values]
    observed = sum(ranks[:len(x)])
    counts = [dict() for _ in range(len(x) + 1)]
    counts[0][0] = 1
    for i, rank in enumerate(ranks):
        for k in range(min(i + 1, len(x)), 0, -1):
            for total, count in list(counts[k - 1].items()):
                if total + rank <= observed:
                    counts[k][total + rank] = counts[k].get(total + rank, 0) + count
    p = sum(counts[len(x)].values()) / math.comb(len(values), len(x))
    u = observed / 2 - len(x) * (len(x) + 1) / 2
    return {"u": u, "p_less": p, "method": "exact_label_permutation_average_rank_ties"}


def _pair_array(pairs, languages, seeds):
    languages, seeds = tuple(languages), tuple(seeds)
    if len(set(languages)) != len(languages) or len(set(seeds)) != len(seeds) or not seeds:
        raise ValueError("Languages and seeds must be unique and nonempty")
    edges = tuple(itertools.combinations(languages, 2))
    index = {frozenset(edge): i for i, edge in enumerate(edges)}
    seed_index = {s: i for i, s in enumerate(seeds)}
    values = np.full((len(seeds), len(edges)), np.nan)
    for row in pairs:
        edge = frozenset((row["left"], row["right"]))
        if row["seed"] not in seed_index or edge not in index or row["left"] == row["right"]:
            raise ValueError("Pair has an unknown seed/language or self edge")
        i, j = seed_index[row["seed"]], index[edge]
        if not np.isnan(values[i, j]):
            raise ValueError("Duplicate language pair within seed")
        value = float(row["auc"])
        if not math.isfinite(value) or value < 0:
            raise ValueError("Pair AUC must be finite and nonnegative")
        values[i, j] = value
    if np.isnan(values).any():
        raise ValueError("Missing language pairs or seeds")
    return edges, values


def panel_summary(pairs, languages=ROMANCE + CONTROLS, seeds=SEEDS, focal=ROMANCE):
    """One contrast per seed; pair dependence is retained inside each contrast."""
    languages, seeds, focal = tuple(languages), tuple(seeds), tuple(focal)
    if len(set(focal)) != len(focal) or not set(focal) <= set(languages):
        raise ValueError("Invalid focal group")
    edges, values = _pair_array(pairs, languages, seeds)
    mask = np.array([a in focal and b in focal for a, b in edges])
    if not mask.any() or mask.all():
        raise ValueError("Panel requires within-focal and comparison pairs")
    focal_means, other_means = values[:, mask].mean(axis=1), values[:, ~mask].mean(axis=1)
    deltas = other_means - focal_means
    per_seed = [{"seed": s, "means": {"romance": float(a), "control": float(b)},
                 "other_minus_romance_auc": float(d), "direction_replicates": bool(d > 1e-12)}
                for s, a, b, d in zip(seeds, focal_means, other_means, deltas)]
    return {"effect": float(deltas.mean()), "per_seed": per_seed,
            "direction_replicates_all_seeds": bool(np.all(deltas > 1e-12)),
            "focal_pairs_per_seed": int(mask.sum()), "other_pairs_per_seed": int((~mask).sum())}


def whole_language_reference(pairs, languages=ROMANCE + CONTROLS, seeds=SEEDS, focal=ROMANCE):
    """Exact allocation reference; p-value validity requires label exchangeability."""
    languages, seeds, focal = tuple(languages), tuple(seeds), tuple(focal)
    observed = panel_summary(pairs, languages, seeds, focal)["effect"]
    edges, values = _pair_array(pairs, languages, seeds)
    allocations = []
    for selected in itertools.combinations(languages, len(focal)):
        mask = np.array([a in selected and b in selected for a, b in edges])
        deltas = values[:, ~mask].mean(axis=1) - values[:, mask].mean(axis=1)
        allocations.append({"focal_languages": list(selected), "effect": float(deltas.mean()),
                            "direction_replicates_all_seeds": bool(np.all(deltas > 1e-12))})
    tolerance = 1e-12 * max(1.0, abs(observed))
    extreme = sum(a["effect"] >= observed - tolerance for a in allocations)
    return {"method": "exhaustive_whole_language_bundles_shared_across_seeds",
            "observed_effect": observed, "extreme_allocations": extreme,
            "allocation_count": len(allocations), "upper_tail_fraction": extreme / len(allocations),
            "tail_tolerance": tolerance, "allocations": allocations,
            "exchangeability_established": False, "unconditional_significance_claim_allowed": False,
            "conditional_null": "Joint trajectories invariant under whole-language label permutations",
            "interpretation": "Descriptive label reference; a valid permutation p-value only under the stated exchangeability null"}


def leave_one_language_out(pairs):
    rows = []
    for omitted in ROMANCE + CONTROLS:
        remaining = tuple(l for l in ROMANCE + CONTROLS if l != omitted)
        filtered = [p for p in pairs if omitted not in (p["left"], p["right"])]
        summary = panel_summary(filtered, remaining, SEEDS, tuple(l for l in ROMANCE if l != omitted))
        rows.append({"omitted_language": omitted, **summary})
    return {"analyses": rows, "effect_range": [min(r["effect"] for r in rows), max(r["effect"] for r in rows)],
            "interpretation": "Sensitivity to panel membership, not confidence intervals or independent tests"}


def sample_deciles(scores, vectors):
    scores, vectors = np.asarray(scores, dtype=float), np.asarray(vectors, dtype=float)
    if (scores.ndim != 1 or vectors.ndim != 2 or len(scores) < 2
            or vectors.shape[0] != len(scores) or vectors.shape[1] == 0
            or not np.isfinite(scores).all() or not np.isfinite(vectors).all()):
        raise ValueError("Trajectory requires at least two finite measured states")
    if scores[-1] <= scores[0]:
        raise ValueError("H1 progress is undefined without positive final structural improvement")
    progress = (scores - scores[0]) / (scores[-1] - scores[0])
    result = [vectors[0]]
    for p in np.arange(0.1, 1.0, 0.1):
        i = next(i for i in range(1, len(scores)) if progress[i - 1] < p <= progress[i])
        fraction = (p - progress[i - 1]) / (progress[i] - progress[i - 1])
        result.append(vectors[i - 1] + fraction * (vectors[i] - vectors[i - 1]))
    result.append(vectors[-1])
    return np.array(result)


def pair_auc(left, right):
    left, right = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if left.ndim != 2 or left.shape != right.shape or left.shape[0] != 11 or left.shape[1] == 0:
        raise ValueError("Pair curves must have matching 11-decile vector shapes")
    distances = np.linalg.norm(left - right, axis=1)
    if distances[0] <= 0 or not np.isfinite(distances).all():
        raise ValueError("H1 pair has zero initial separation or nonfinite distance")
    normalized = distances / distances[0]
    return float(np.trapezoid(normalized, dx=0.1)), normalized.tolist()


def load_trajectory(path, *, require_progress=True):
    from src.retrodiction.h1_objective import H1Objective, text_hash
    manifest = json.loads(path.read_text(encoding="utf-8"))
    execution = manifest.get("h1_execution")
    if not execution:
        raise ValueError(f"Unverified/legacy run: {path}")
    cfg = manifest["config"]
    if cfg["total_target_proposals"] != 0 or manifest["status"] not in {"joint_hit", "plateau_hit"}:
        raise ValueError(f"Bounded pilot or unfinished run cannot decide H1: {path}")
    obj = H1Objective.load(cfg["features_freeze_commit"], cfg["features_freeze_sha256"], cfg["corpus_budget"])
    if obj.provenance != execution["objective"]:
        raise ValueError(f"H1 implementation/calibration identity mismatch: {path}")
    scores, vectors = [], []
    previous_checkpoint = None
    for block in manifest["blocks"]:
        summary = json.loads(Path(block["summary_path"]).read_text(encoding="utf-8"))
        audit = summary.get("h1_audit", {})
        if audit.get("identity_sha256") != obj.provenance["identity_sha256"]:
            raise ValueError("Unaudited H1 block")
        checkpoint = Path(summary["best_corpus_json"])
        if text_hash(checkpoint) != audit["checkpoint_sha256_lf"]:
            raise ValueError("H1 checkpoint changed since audit")
        for i, stage in enumerate(summary["stages"]):
            seqs = json.loads(Path(stage["artifacts"]["corpus_json"]).read_text(encoding="utf-8"))["sequences"]
            measured = obj.evaluate(seqs)
            stored = stage["diagnostics"]["h1"]
            if (stored["identity_sha256"] != obj.provenance["identity_sha256"]
                    or not np.allclose(measured["z_6d"], stored["z_6d"], atol=1e-10, rtol=0)
                    or abs(measured["structural_score"] - stored["structural_score"]) > 1e-10):
                raise ValueError("H1 trajectory replay failed")
            if i == 0 and previous_checkpoint is not None:
                if seqs != previous_checkpoint:
                    raise ValueError("H1 corpus changed at block boundary")
                continue
            scores.append(measured["structural_score"])
            vectors.append(measured["z_6d"])
        previous_checkpoint = json.loads(checkpoint.read_text(encoding="utf-8"))["sequences"]
    if not scores:
        raise ValueError(f"Run has no measured states: {path}")
    return manifest, sample_deciles(scores, vectors) if require_progress else None


def analyze(root: Path):
    root = root.resolve()
    for language in ROMANCE + CONTROLS:
        for seed in SEEDS:
            path = root / language / f"seed_{seed}" / "manifest.json"
            if not path.exists():
                raise ValueError(f"Missing required H1 trajectory: {path}")
    registration = validate_registration(root)
    trajectories = {}
    identities = set()
    contracts = set()
    diagnostics, inputs = [], []
    for language in ROMANCE + CONTROLS:
        for seed in SEEDS:
            path = root / language / f"seed_{seed}" / "manifest.json"
            if not path.exists():
                raise ValueError(f"Missing required H1 trajectory: {path}")
            manifest, deciles = load_trajectory(path)
            validate_registered_manifest(registration, manifest)
            if manifest["config"]["language"] != language or manifest["config"]["seed"] != seed:
                raise ValueError("Matrix directory and manifest language/seed disagree")
            identities.add(manifest["h1_execution"]["objective"]["identity_sha256"])
            contract = dict(manifest["h1_execution"]["launch_contract"])
            for key in ("language", "seed", "start_corpus"):
                contract.pop(key, None)
            contracts.add(json.dumps(contract, sort_keys=True))
            trajectories[language, seed] = deciles
            diagnostics.append(trajectory_diagnostics(manifest))
            from src.retrodiction.h1_objective import text_hash
            inputs.append({"path": str(path), "manifest_sha256_lf": text_hash(path)})
    if len(identities) != 1 or len(contracts) != 1:
        raise ValueError("H1 matrix mixes different objectives or launch conditions")
    pairs = []
    for seed in SEEDS:
        for a, b in itertools.combinations(ROMANCE + CONTROLS, 2):
            auc, curve = pair_auc(trajectories[a, seed], trajectories[b, seed])
            group = "romance" if a in ROMANCE and b in ROMANCE else "control"
            pairs.append({"seed": seed, "left": a, "right": b, "group": group,
                          "auc": auc, "normalized_distance_deciles": curve})
    primary = panel_summary(pairs)
    reference = whole_language_reference(pairs)
    stress = []
    for language in ("markov_null", "klingon"):
        for seed in SEEDS:
            path = root / language / f"seed_{seed}" / "manifest.json"
            if not path.exists():
                stress.append({"language": language, "seed": seed, "status": "missing"})
                continue
            try:
                manifest, _ = load_trajectory(path, require_progress=False)
                validate_registered_manifest(registration, manifest)
                stress.append({"language": language, "seed": seed, "status": "complete",
                               "diagnostics": trajectory_diagnostics(manifest)})
                inputs.append({"path": str(path), "manifest_sha256_lf": text_hash(path)})
            except (ValueError, KeyError, FileNotFoundError) as exc:
                stress.append({"language": language, "seed": seed, "status": "not_evaluable", "reason": str(exc)})
    from src.retrodiction.h1_objective import text_hash
    return {
        "schema": "h1_e1_fixed_panel_v2", "objective_identity": next(iter(identities)),
        "study_registration_sha256": registration["registration_sha256"],
        "analysis_source_sha256_lf": text_hash(Path(__file__)),
        "gate": "DESCRIPTIVE_ONLY", "h1_inference_status": "not_established_by_this_design",
        "observed_pattern": "CONSISTENT_ROMANCE_DIRECTION" if primary["direction_replicates_all_seeds"] else "NO_CONSISTENT_ROMANCE_DIRECTION",
        "primary": primary, "language_label_reference": reference,
        "leave_one_language_out": leave_one_language_out(pairs), "pairs": pairs,
        "primary_trajectory_diagnostics": diagnostics, "stress_diagnostics": stress,
        "stress_complete": all(s["status"] == "complete" for s in stress), "inputs": inputs,
        "interpretation": "Decision 046 fixed-panel comparison; no unconditional significance or ancestry verdict",
        "limitations": ["Whole-language label exchangeability is not established for this selected corpus panel.",
                        "Seeds do not create independent language samples; leave-one-out ranges are not confidence intervals.",
                        "Random-init robustness, other budgets and E2-E4 are deferred; full original H1 is not completed."],
    }


def trajectory_diagnostics(manifest):
    from src.retrodiction.h1_objective import H1Objective
    cfg = manifest["config"]
    obj = H1Objective.load(cfg["features_freeze_commit"], cfg["features_freeze_sha256"], cfg["corpus_budget"])
    first = json.loads(Path(manifest["blocks"][0]["summary_path"]).read_text(encoding="utf-8"))
    last = json.loads(Path(manifest["blocks"][-1]["summary_path"]).read_text(encoding="utf-8"))
    before_seqs = json.loads(Path(first["stages"][0]["artifacts"]["corpus_json"]).read_text(encoding="utf-8"))["sequences"]
    after_seqs = json.loads(Path(last["best_corpus_json"]).read_text(encoding="utf-8"))["sequences"]
    before, after = obj.evaluate(before_seqs), obj.evaluate(after_seqs)
    return {"language": cfg["language"], "seed": cfg["seed"], "initial": before, "final": after,
            "feature_changes": {k: after["features"][k] - v for k, v in before["features"].items()},
            "endpoint_bank_distances": obj.reference_distances(after_seqs)}


def render_report(result):
    lines = ["# Corrected E1 fixed-panel readout", "",
             "Question: do the five selected Romance corpora converge toward one another more strongly",
             "than pairs involving the four specified controls, across matched normalized score progress?", ""]
    if result.get("status") == "not_evaluable":
        return "\n".join(lines + ["Status: not evaluable.", "", result["error"], "",
                                  "This is an analysis/collection failure, not a negative scientific answer.", ""])
    primary, ref = result["primary"], result["language_label_reference"]
    lines += [f"Observed pattern: {result['observed_pattern']}.", "",
              f"Mean other-minus-Romance AUC: {primary['effect']:.9g}. Positive values favor greater Romance convergence.", "",
              "| Seed | Romance mean AUC | Other mean AUC | Difference |", "|---|---:|---:|---:|"]
    lines += [f"| {s['seed']} | {s['means']['romance']:.9g} | {s['means']['control']:.9g} | {s['other_minus_romance_auc']:.9g} |"
              for s in primary["per_seed"]]
    lines += ["", f"Whole-language reference tail: {ref['extreme_allocations']}/{ref['allocation_count']} = {ref['upper_tail_fraction']:.9g}.",
              "This fraction has a p-value interpretation only under whole-language label exchangeability,",
              "which is not established for this deliberately selected panel. It is not an unconditional H1 significance verdict.", "",
              f"Separate stress diagnostics complete: {result['stress_complete']}.", "",
              "Leave-one-language-out effects (sensitivity, not confidence intervals):", ""]
    lines += [f"- Omit {r['omitted_language']}: {r['effect']:.9g}; consistent direction: {r['direction_replicates_all_seeds']}."
              for r in result["leave_one_language_out"]["analyses"]]
    lines += ["", *result["limitations"], "", f"Registration: `{result['study_registration_sha256']}`.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report_path = args.output.with_suffix(".md")
    if args.output.suffix.lower() != ".json" or args.output.exists() or report_path.exists():
        parser.error("Use a fresh .json output path (the matching .md report must also be new)")
    try:
        result = analyze(args.run_root)
    except (ValueError, KeyError, FileNotFoundError) as exc:
        result = {"schema": "h1_e1_fixed_panel_v2", "status": "not_evaluable", "gate": "NOT_EVALUABLE",
                  "h1_inference_status": "not_evaluable", "error": str(exc),
                  "interpretation": "Analysis/collection failure, not a negative scientific result"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, allow_nan=False)
    report_path.write_text(render_report(result), encoding="utf-8")
    if result.get("status") == "not_evaluable":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
