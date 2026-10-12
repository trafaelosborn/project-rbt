"""Replay every saved state of completed, finite H1 benchmark jobs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.retrodiction.h1_objective import H1Objective, text_hash
from src.validation.h1_convergence import load_trajectory


def verify(root):
    results = json.loads((root / "benchmark_results.json").read_text(encoding="utf-8"))
    checked = []
    for result in results:
        if result["status"] != "completed":
            checked.append({"job_id": result["job_id"], "status": "incomplete_not_replayed"})
            continue
        path = Path(result["manifest"])
        before_hash = text_hash(path)
        manifest = json.loads(path.read_text(encoding="utf-8"))
        cfg = manifest["config"]
        if cfg["total_target_proposals"] <= 0 or cfg["n_candidates"] != 16:
            raise AssertionError("Expected a finite, 16-candidate benchmark")
        obj = H1Objective.load(cfg["features_freeze_commit"], cfg["features_freeze_sha256"], cfg["corpus_budget"])
        if manifest["h1_execution"]["objective"] != obj.provenance:
            raise AssertionError("Benchmark objective identity mismatch")
        stages, max_error, proposals = 0, 0.0, 0
        previous = None
        for block in manifest["blocks"]:
            summary = json.loads(Path(block["summary_path"]).read_text(encoding="utf-8"))
            proposals += summary["proposals_attempted"]
            for i, stage in enumerate(summary["stages"]):
                seqs = json.loads(Path(stage["artifacts"]["corpus_json"]).read_text(encoding="utf-8"))["sequences"]
                if i == 0 and previous is not None and seqs != previous:
                    raise AssertionError("Block continuity mismatch")
                measured = obj.evaluate(seqs)
                saved = stage["diagnostics"]["h1"]
                if measured["measured_token_count"] != 20000 or saved["identity_sha256"] != obj.provenance["identity_sha256"]:
                    raise AssertionError("Budget or stage identity mismatch")
                for key, value in measured["features"].items():
                    if abs(value - saved["features"][key]) > 1e-10:
                        raise AssertionError(f"Feature replay failed: {key}")
                if abs(measured["structural_score"] - saved["structural_score"]) > 1e-10:
                    raise AssertionError("Full-precision score replay failed")
                error = abs(measured["structural_score"] - stage["latin_structural_score"])
                if error > 1e-6:
                    raise AssertionError("Rounded score replay failed")
                max_error = max(max_error, error)
                stages += 1
            checkpoint = Path(summary["best_corpus_json"])
            if text_hash(checkpoint) != summary["h1_audit"]["checkpoint_sha256_lf"]:
                raise AssertionError("Checkpoint hash mismatch")
            previous = json.loads(checkpoint.read_text(encoding="utf-8"))["sequences"]
        if proposals != manifest["completed_block_proposals"]:
            raise AssertionError("Proposal count mismatch")
        try:
            load_trajectory(path)
        except ValueError as exc:
            if "Bounded pilot" not in str(exc):
                raise
        else:
            raise AssertionError("Benchmark was admitted as a primary scientific run")
        if text_hash(path) != before_hash:
            raise AssertionError("Verification changed the benchmark manifest")
        checked.append({"job_id": result["job_id"], "status": "verified",
                        "stages_replayed": stages, "max_rounded_score_error": max_error,
                        "actual_proposals": proposals, "manifest_sha256_lf": before_hash,
                        "scientific_readout_excluded": True})
    return {"schema": "h1_benchmark_replay_v1", "results": checked,
            "script_sha256_lf": text_hash(Path(__file__))}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    result = verify(args.root.resolve())
    with (args.root / "verification.json").open("x", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    print(json.dumps(result, indent=2))
