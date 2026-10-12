"""Fail-closed H1 launch/resume checks and independently replayed block audits."""
from __future__ import annotations

import json
import csv
from pathlib import Path

import numpy as np

from src.retrodiction.h1_objective import H1Objective, PROJECT_ROOT, stable_hash, text_hash


def prepare_execution(config, manifest: dict | None):
    objective = H1Objective.from_config(config)
    old_config = (manifest or {}).get("config", {})
    if objective is None:
        if manifest and (manifest.get("h1_execution") or old_config.get("features_freeze_commit")
                         or old_config.get("features_freeze_sha256")):
            raise ValueError("Cannot resume a frozen run through the legacy scorer")
        return None, None
    if manifest and not manifest.get("h1_execution"):
        raise ValueError("Legacy manifest has no verified H1 objective; use a fresh output directory")
    if config.starting_proposals != 0:
        raise ValueError("Corrected H1 must start from its source corpus, with starting_proposals=0")
    bank = json.loads((PROJECT_ROOT / "data/v2/corpora_manifest.json").read_text(encoding="utf-8"))
    entries = [r for r in bank["corpora"] if r["language"] == config.language and r["status"] == "ready"]
    source = config.start_corpus.resolve()
    if not any(source == (PROJECT_ROOT / r["tokens_path"]).resolve() for r in entries):
        raise ValueError("H1 start corpus must be the frozen bank source for this language")
    sequences = json.loads(source.read_text(encoding="utf-8"))["sequences"]
    objective.measure(sequences)  # fail before creating output directories
    if config.validator_set:
        allowed = {r["corpus_id"] for r in objective.rows}
        if not set(config.validator_set) <= allowed:
            raise ValueError("H1 validator IDs must belong to the frozen calibration bank")
    launch = config.to_dict()
    for key in ("output_dir", "total_target_proposals", "live_event_mode", "live_event_buffer_size"):
        launch.pop(key, None)
    launch["start_corpus"] = str(source)
    execution = {
        "objective": objective.provenance,
        "source_sha256_lf": text_hash(source),
        "launch_contract": launch,
        "analysis_status": "trajectory_collection_only_not_an_H1_gate_result",
    }
    execution["execution_sha256"] = stable_hash(execution)
    if manifest:
        if manifest["h1_execution"] != execution:
            raise ValueError("H1 resume contract changed (objective/code/source/config); use a new run")
        blocks = manifest.get("blocks", [])
        if blocks:
            checkpoint = Path(manifest["current_corpus"])
            expected = blocks[-1].get("h1_audit", {}).get("checkpoint_sha256_lf")
            if not expected or text_hash(checkpoint) != expected:
                raise ValueError("H1 checkpoint is missing an audit or changed since its block audit")
    return objective, execution


def audit_block(objective, config, summary: dict, summary_path: Path, engine) -> dict:
    """Recompute saved features/scores; never merely trust manifest metadata."""
    from src.validation.static_feature_matrix import compute_feature_vector

    stages = summary.get("stages", [])
    if not stages:
        raise ValueError("H1 block has no auditable stages")
    max_error = 0.0
    for stage in stages:
        h1 = stage.get("diagnostics", {}).get("h1")
        if not h1 or h1.get("identity_sha256") != objective.provenance["identity_sha256"]:
            raise ValueError("Block stage was not evaluated by the active H1 objective")
        path = Path(stage["artifacts"]["corpus_json"])
        seqs = json.loads(path.read_text(encoding="utf-8"))["sequences"]
        measured = objective.measure(seqs)
        features = compute_feature_vector(measured)
        for name, value in features.items():
            if not np.isclose(value, h1["features"][name], rtol=0, atol=1e-10):
                raise ValueError(f"H1 persisted feature mismatch: {stage['stage_id']} / {name}")
        raw = np.array([features[c] for c in objective.features])
        distance = np.linalg.norm((raw - objective.latin_raw[objective.indices]) / objective.scales[objective.indices])
        expected = float(-objective.score_scale * distance)
        error = abs(expected - stage["latin_structural_score"])
        max_error = max(max_error, error)
        if error > 1e-6 or abs(expected - h1["structural_score"]) > 1e-10:
            raise ValueError(f"H1 saved score failed independent replay: {stage['stage_id']}")
        if h1["measured_token_count"] != objective.budget or h1["state_token_count"] != sum(map(len, seqs)):
            raise ValueError("H1 saved token counts disagree with corpus")
        components = h1["score_components"]
        if abs(sum(components.values()) - h1["base_total_score"]) > 1e-10:
            raise ValueError("H1 composite components do not sum to the recorded base score")
        diagnostics = stage["diagnostics"]
        effective = h1["base_total_score"] + diagnostics.get("reward_bonus", 0) + diagnostics.get("reward_penalty_relief", 0)
        if abs(effective - stage["total_score"]) > 1e-6:
            raise ValueError("H1 effective score does not reproduce saved reward amplification")

    checkpoint = Path(summary["best_corpus_json"])
    final = stages[-1]
    # Acceptance increases effective total score, so the continued state must
    # agree with the final stage used by stop/progress summaries.
    if checkpoint != Path(final["artifacts"]["corpus_json"]):
        raise ValueError("H1 best checkpoint differs from the reported final state")
    audit = {
        "objective_id": objective.provenance["objective_id"],
        "identity_sha256": objective.provenance["identity_sha256"],
        "stages_replayed": len(stages), "max_rounded_score_error": max_error,
        "checkpoint_sha256_lf": text_hash(checkpoint),
        "measured_token_count": objective.budget,
        "invalid_budget_candidates": getattr(engine, "_h1_budget_rejections", 0),
        "requested_incremental_scoring": config.use_incremental_scoring,
        "requested_fortran_batch": config.use_fortran_batch,
        "effective_candidate_scoring_backend": "h1_python_reference",
    }
    summary["h1_audit"] = audit
    summary["h1_objective"] = objective.provenance
    summary["final_h1_measurement"] = final["diagnostics"]["h1"]
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    seqs = json.loads(checkpoint.read_text(encoding="utf-8"))["sequences"]
    (summary_path.parent / "h1_reference_distances.json").write_text(json.dumps({
        "objective_identity": objective.provenance["identity_sha256"],
        "corpus_budget": objective.budget,
        "distances": objective.reference_distances(seqs),
        "interpretation": "diagnostic endpoint distances; not the H1 trajectory convergence gate",
    }, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return summary


def compare_saved_run(manifest_path, output_prefix=None, validator_ids=None, output_dir=None, block_ids=None):
    """Compatibility entry point for controller/validator-bank commands on H1."""
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    cfg = manifest["config"]
    objective = H1Objective.load(cfg["features_freeze_commit"], cfg["features_freeze_sha256"], cfg["corpus_budget"])
    if objective.provenance != manifest["h1_execution"]["objective"]:
        raise ValueError("Saved H1 objective differs from this implementation")
    blocks = [b for b in manifest["blocks"] if block_ids is None or b["block"] in block_ids]
    if not blocks:
        raise ValueError("No H1 blocks selected")
    if validator_ids is not None and not set(validator_ids) <= {r["corpus_id"] for r in objective.rows}:
        raise ValueError("H1 comparison requested unknown frozen-bank validator IDs")
    rows, nearest = [], []
    for block in blocks:
        path = Path(block["best_corpus_json"])
        if text_hash(path) != block.get("h1_audit", {}).get("checkpoint_sha256_lf"):
            raise ValueError("H1 endpoint changed or lacks an audit")
        seqs = json.loads(path.read_text(encoding="utf-8"))["sequences"]
        distances = objective.reference_distances(seqs)
        if validator_ids is not None:
            distances = [r for r in distances if r["corpus_id"] in validator_ids]
        if not distances:
            raise ValueError("No H1 references selected")
        rows.extend({"block_id": block["block"], **r} for r in distances)
        best = min(distances, key=lambda r: r["euclidean_6d"])
        nearest.append({"block_id": block["block"], "validator_corpus": best["corpus_id"],
                        "validator_structural_distance": best["euclidean_6d"]})
    directory = Path(output_dir) if output_dir is not None else Path(manifest_path).parent / "h1_validation"
    directory.mkdir(parents=True, exist_ok=True)
    prefix = output_prefix or "h1_reference_comparison"
    json_path, csv_path = directory / f"{prefix}.json", directory / f"{prefix}.csv"
    result = {"objective": objective.provenance, "rows": rows,
              "summary": {"nearest_structural_by_block": nearest},
              "interpretation": "Frozen H1 endpoint geometry; not a convergence-gate verdict",
              "validator_count": len(distances), "block_count": len(blocks),
              "json_path": str(json_path), "csv_path": str(csv_path), "summary_path": str(json_path)}
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return result
