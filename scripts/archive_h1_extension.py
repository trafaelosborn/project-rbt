"""Archive the validated Decision 047 implementation and evidence identities."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.control.h1_ablation import identity, validate_registration
from src.retrodiction.h1_checkpointed import read_json
from src.retrodiction.h1_objective import text_hash


def main():
    root = ROOT / "artifacts/h1_extension_20261001"
    for name in ("pilot_01", "process_resume_01"):
        if not read_json(root / name / "verification_results.json").get("complete"):
            raise ValueError(f"Incomplete validation: {name}")
    # The exact validated implementation must still be active before archival.
    validate_registration(root / "pilot_01/paused")
    validate_registration(root / "process_resume_01/split")
    report = ET.parse(root / "tests_03.xml")
    cases = list(report.iter("testcase"))
    if any(any(case.find(tag) is not None for tag in ("failure", "error", "skipped")) for case in cases):
        raise ValueError("Broad suite has failed/skipped tests")
    files = {p for folder in ("src", "tests", "scripts") for p in (ROOT / folder).rglob("*.py")}
    files.update((ROOT / "src").rglob("*.f90"))
    for relative in (
        "README.md", "requirements.txt", "docs/features_v2.md",
        "docs/decisions/044_h1_objective_execution_repair.md",
        "docs/decisions/045_h1_readout_operational_clarifications.md",
        "docs/decisions/046_h1_e1_dependence_and_claim_scope.md",
        "docs/decisions/047_h1_continuous_search_and_form_reward_ablation.md",
        "docs/registrations/h1_e1_panel_v2.json", "docs/registrations/h1_form_reward_ablation_v1.json",
        "docs/reports/2026-10-01_h1_parameter_review.md",
        "docs/reports/2026-10-01_h1_extension_audit.md",
        "docs/reports/2026-10-01_h1_design_repair_audit.md",
        "data/v2/static_feature_matrix.csv", "data/v2/static_feature_tiers.json", "data/v2/corpora_manifest.json",
    ):
        files.add(ROOT / relative)
    archive = root / "source_snapshot.zip"
    hashes = {}
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as destination:
        for path in sorted(files):
            relative = path.relative_to(ROOT).as_posix()
            destination.write(path, relative)
            hashes[relative] = text_hash(path)
    receipt = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "implementation_identity": identity(),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_hashes_lf": hashes, "snapshot_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "broad_suite_passing_tests": len(cases), "production_launched": False,
        "evidence_files_sha256": {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in sorted(root.rglob("*")) if p.is_file() and p != archive},
        "interpretation": "Implementation validation only. Preserves pre-existing code; raw source corpora stay at registered paths.",
    }
    with (root / "completion_inventory.json").open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
    print(json.dumps({key: receipt[key] for key in ("snapshot_sha256", "broad_suite_passing_tests", "production_launched")}, indent=2))


if __name__ == "__main__":
    main()
