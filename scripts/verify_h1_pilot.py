"""Read/replay the bounded H1 pilot and verify completed-run idempotence.

Usage: python scripts/verify_h1_pilot.py <pilot-root>
Writes a new verification.json alongside the trial, preserving run artifacts.
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.control.h1_matrix import build_jobs
from src.retrodiction.h1_objective import H1Objective, FREEZE_COMMIT, FEATURE_SHA256, text_hash
from src.retrodiction.h1_execution import prepare_execution
from src.retrodiction.long_run_v4 import LongRunConfig
from src.validation.h1_convergence import load_trajectory


def verify(root):
    objective = H1Objective.load(FREEZE_COMMIT, FEATURE_SHA256, 20000)
    results = []
    for job in build_jobs(root, pilot=True):
        manifest = json.loads(job.manifest_path.read_text(encoding="utf-8"))
        assert manifest["status"] == "complete"
        assert manifest["h1_execution"]["objective"] == objective.provenance
        last = None
        stage_count = 0
        max_error = 0.0
        accepted = 0
        start_score = None
        for block in manifest["blocks"]:
            summary = json.loads(Path(block["summary_path"]).read_text(encoding="utf-8"))
            for i, stage in enumerate(summary["stages"]):
                sequences = json.loads(Path(stage["artifacts"]["corpus_json"]).read_text(encoding="utf-8"))["sequences"]
                replay = objective.evaluate(sequences)
                error = abs(replay["structural_score"] - stage["latin_structural_score"])
                assert error < 1e-6
                assert replay["measured_token_count"] == 20000
                if i == 0 and last is not None:
                    assert sequences == last, "Corpus changed at the block boundary"
                if start_score is None:
                    start_score = replay["structural_score"]
                max_error = max(max_error, error)
                stage_count += 1
            last = sequences
            accepted += summary["accepted_mutation_stages"]
            comparison = json.loads((Path(block["summary_path"]).parent / "h1_reference_distances.json").read_text())
            latin = next(r for r in comparison["distances"] if r["language"] == "latin")
            assert abs(latin["euclidean_6d"] + replay["structural_score"] / 5) < 1e-10
        hashes_before = {str(p): text_hash(p) for p in job.output_dir.rglob("*.json")}
        with (root / f"{job.job_id}_completed_resume.log").open("w", encoding="utf-8") as output:
            subprocess.run(job.command(), cwd=ROOT, check=True, stdout=output, stderr=subprocess.STDOUT)
        hashes_after = {str(p): text_hash(p) for p in job.output_dir.rglob("*.json")}
        assert hashes_before == hashes_after, "Completed resume modified run artifacts"
        try:
            load_trajectory(job.manifest_path)
        except ValueError as exc:
            assert "Bounded pilot" in str(exc)
        else:
            raise AssertionError("Pilot incorrectly admitted as an H1 gate trajectory")
        results.append({"language": job.language, "seed": job.seed, "blocks": len(manifest["blocks"]),
                        "proposals": manifest["cumulative_proposals"], "accepted_stages": accepted,
                        "stages_replayed": stage_count, "max_rounded_score_error": max_error,
                        "start_structural_score": start_score, "final_structural_score": replay["structural_score"],
                        "measured_tokens": 20000, "block_continuity": "exact",
                        "completed_resume": "no_run_artifacts_changed", "gate_exclusion": "verified"})
    historical = ROOT / "data/v2_h1_production_20260613/french/seed_42/manifest.json"
    before = text_hash(historical)
    old = json.loads(historical.read_text(encoding="utf-8"))
    cfg = LongRunConfig("french", ROOT / "data/processed/romance/french_tokens.json", historical.parent,
                        features_freeze_commit=FREEZE_COMMIT, features_freeze_sha256=FEATURE_SHA256)
    try:
        prepare_execution(cfg, old)
    except ValueError as exc:
        assert "fresh output directory" in str(exc)
    else:
        raise AssertionError("Historical production manifest incorrectly admitted")
    assert text_hash(historical) == before
    result = {"objective_identity": objective.provenance["identity_sha256"], "runs": results,
              "historical_manifest_guard": "rejected_without_modification", "h1_verdict": "not_tested"}
    (root / "verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(verify(Path(sys.argv[1]).resolve()), indent=2))
