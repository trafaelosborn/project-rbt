"""Held-out control comparison for the completed v2 H1 production matrix.

This is a read-only, post-run analysis.  It applies the first paper's
control-bank logic to every production start/final pair, using the frozen v2
feature definitions and a common token budget.  Generated corpora are never
used to estimate the reference-space mean or scale.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from src.validation.checkpoint_compare import CorpusFormReference
from src.validation.static_feature_matrix import (
    DEFAULT_MANIFEST_PATH,
    REGISTERED_FEATURES,
    _load_manifest,
    _load_sequences_for_entry,
    compute_feature_vector,
    select_manifest_entries,
    subsample_sequences,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_ROOT = PROJECT_ROOT / "data" / "v2_h1_production_20260613"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "validation" / "v2_h1_endpoint_controls_20260804"
DEFAULT_TOKEN_BUDGET = 9_000
DEFAULT_SAMPLE_SEED = 1729

CONTROL_LANGUAGES = (
    "latin", "markov_null", "finnish", "turkish", "vietnamese", "sumerian", "klingon"
)


def _read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _sample(sequences: list[list[str]], *, corpus_id: str, budget: int, seed: int) -> list[list[str]]:
    return subsample_sequences(sequences, token_budget=budget, seed=seed, corpus_id=corpus_id)


def _production_states(run_root: Path, budget: int, seed: int) -> list[dict]:
    states: list[dict] = []
    for manifest_path in sorted(run_root.glob("*/seed_*/manifest.json")):
        manifest = _read_json(manifest_path)
        language = manifest["config"]["language"]
        run_seed = int(manifest["config"]["seed"])
        first = manifest["blocks"][0]
        record_dir = Path(first["summary_path"]).parent / "records"
        initial_record = sorted(record_dir.glob("*_v5_000.json"))[0]
        initial_path = Path(_read_json(initial_record)["artifacts"]["corpus_json"])
        final_path = Path(manifest["current_corpus"])
        for state, corpus_path in (("start", initial_path), ("final", final_path)):
            sequences = _read_json(corpus_path)["sequences"]
            corpus_id = f"{language}_seed_{run_seed}_{state}"
            sampled = _sample(sequences, corpus_id=corpus_id, budget=budget, seed=seed)
            states.append({
                "corpus_id": corpus_id,
                "language": language,
                "seed": run_seed,
                "state": state,
                "path": str(corpus_path),
                "sequences": sampled,
                "features": compute_feature_vector(sampled),
            })
    return states


def _reference_rows(manifest_path: Path, budget: int, seed: int) -> list[dict]:
    manifest = _load_manifest(manifest_path)
    rows: list[dict] = []
    for entry in select_manifest_entries(manifest, include_optional=True):
        sequences = _load_sequences_for_entry(entry)
        sampled = _sample(sequences, corpus_id=entry["corpus_id"], budget=budget, seed=seed)
        rows.append({
            "corpus_id": entry["corpus_id"],
            "language": entry["language"],
            "role": entry["role"],
            "family": entry.get("family", ""),
            "sequences": sampled,
            "features": compute_feature_vector(sampled),
        })
    return rows


def _cosine_distance(left: np.ndarray, right: np.ndarray) -> float:
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom == 0.0:
        return 0.0
    return float(1.0 - np.dot(left, right) / denom)


def run_analysis(run_root: Path, manifest_path: Path, output_dir: Path, budget: int, seed: int) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    references = _reference_rows(manifest_path, budget, seed)
    states = _production_states(run_root, budget, seed)
    feature21 = [spec.column for spec in REGISTERED_FEATURES]
    tiers = _read_json(PROJECT_ROOT / "data" / "v2" / "static_feature_tiers.json")
    feature6 = list(tiers["retained_in_loop"])

    reference_matrix = np.array(
        [[row["features"][col] for col in feature21] for row in references], dtype=np.float64
    )
    means = reference_matrix.mean(axis=0)
    scales = reference_matrix.std(axis=0, ddof=0)
    scales[scales == 0.0] = 1.0
    controls = [row for row in references if row["language"] in CONTROL_LANGUAGES]
    control_forms = {
        row["language"]: CorpusFormReference.from_sequences(row["language"], row["sequences"])
        for row in controls
    }

    feature_rows: list[dict] = []
    for row_type, rows in (("reference", references), ("generated", states)):
        for row in rows:
            feature_rows.append({
                "row_type": row_type,
                "corpus_id": row["corpus_id"],
                "language": row["language"],
                "seed": row.get("seed", ""),
                "state": row.get("state", "reference"),
                **{col: row["features"][col] for col in feature21},
            })

    distances: list[dict] = []
    indexes = {col: feature21.index(col) for col in feature21}
    for state in states:
        raw_state = np.array([state["features"][col] for col in feature21], dtype=np.float64)
        z_state = (raw_state - means) / scales
        for control in controls:
            raw_control = np.array([control["features"][col] for col in feature21], dtype=np.float64)
            z_control = (raw_control - means) / scales
            form = control_forms[control["language"]].score(state["sequences"])
            record = {
                "language": state["language"],
                "seed": state["seed"],
                "state": state["state"],
                "control": control["language"],
                "control_role": control["role"],
                "cosine_distance_6d": _cosine_distance(
                    z_state[[indexes[col] for col in feature6]],
                    z_control[[indexes[col] for col in feature6]],
                ),
                "euclidean_distance_6d": float(np.linalg.norm(
                    z_state[[indexes[col] for col in feature6]] - z_control[[indexes[col] for col in feature6]]
                )),
                "cosine_distance_21d": _cosine_distance(z_state, z_control),
                "euclidean_distance_21d": float(np.linalg.norm(z_state - z_control)),
                "form_score": form["validator_form_score"],
                "char_bigram_cosine": form["validator_char_bigram_cosine"],
                "char_trigram_cosine": form["validator_char_trigram_cosine"],
                "suffix_cosine": form["validator_suffix_cosine"],
            }
            distances.append(record)

    starts = {(r["language"], r["seed"], r["control"]): r for r in distances if r["state"] == "start"}
    movement: list[dict] = []
    for final in (r for r in distances if r["state"] == "final"):
        start = starts[(final["language"], final["seed"], final["control"])]
        movement.append({
            "language": final["language"],
            "seed": final["seed"],
            "control": final["control"],
            "start_cosine_6d": start["cosine_distance_6d"],
            "final_cosine_6d": final["cosine_distance_6d"],
            "delta_cosine_6d": final["cosine_distance_6d"] - start["cosine_distance_6d"],
            "start_cosine_21d": start["cosine_distance_21d"],
            "final_cosine_21d": final["cosine_distance_21d"],
            "delta_cosine_21d": final["cosine_distance_21d"] - start["cosine_distance_21d"],
            "start_form_score": start["form_score"],
            "final_form_score": final["form_score"],
            "delta_form_score": final["form_score"] - start["form_score"],
        })

    nearest: list[dict] = []
    for language in sorted({row["language"] for row in states}):
        for run_seed in sorted({row["seed"] for row in states if row["language"] == language}):
            final_rows = [r for r in distances if r["language"] == language and r["seed"] == run_seed and r["state"] == "final" and r["control"] != "latin"]
            nearest.append({
                "language": language,
                "seed": run_seed,
                "nearest_6d": min(final_rows, key=lambda r: r["cosine_distance_6d"])["control"],
                "nearest_6d_distance": min(r["cosine_distance_6d"] for r in final_rows),
                "nearest_21d": min(final_rows, key=lambda r: r["cosine_distance_21d"])["control"],
                "nearest_21d_distance": min(r["cosine_distance_21d"] for r in final_rows),
                "nearest_euclidean_6d": min(final_rows, key=lambda r: r["euclidean_distance_6d"])["control"],
                "nearest_euclidean_6d_distance": min(r["euclidean_distance_6d"] for r in final_rows),
                "nearest_euclidean_21d": min(final_rows, key=lambda r: r["euclidean_distance_21d"])["control"],
                "nearest_euclidean_21d_distance": min(r["euclidean_distance_21d"] for r in final_rows),
                "nearest_form": max(final_rows, key=lambda r: r["form_score"])["control"],
                "nearest_form_score": max(r["form_score"] for r in final_rows),
            })

    _write_csv(output_dir / "features.csv", feature_rows)
    _write_csv(output_dir / "distances.csv", distances)
    _write_csv(output_dir / "movement.csv", movement)
    _write_csv(output_dir / "nearest_controls.csv", nearest)
    metadata = {
        "schema": "project_rbt_v2_h1_endpoint_control_analysis_v1",
        "run_root": str(run_root),
        "manifest_path": str(manifest_path),
        "token_budget": budget,
        "sample_seed": seed,
        "reference_scaling_policy": "mean/std from frozen attested reference bank only; generated rows excluded",
        "feature_6d": feature6,
        "feature_21d": feature21,
        "production_state_count": len(states),
        "reference_count": len(references),
        "control_languages": list(CONTROL_LANGUAGES),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--token-budget", type=int, default=DEFAULT_TOKEN_BUDGET)
    parser.add_argument("--sample-seed", type=int, default=DEFAULT_SAMPLE_SEED)
    args = parser.parse_args()
    print(json.dumps(run_analysis(args.run_root, args.manifest, args.output_dir, args.token_budget, args.sample_seed), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
