"""Archive the final source/protocol state without replacing earlier evidence."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import xml.etree.ElementTree as ET
import zipfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.retrodiction.h1_objective import PROJECT_ROOT, H1Objective, FREEZE_COMMIT, FEATURE_SHA256, text_hash
from src.validation.h1_protocol import protocol_identity, reference_input_hashes


def archive():
    root = PROJECT_ROOT / "artifacts/h1_design_20261001"
    destination = root / "source_snapshot.zip"
    if destination.exists():
        raise ValueError("Do not replace an existing source snapshot")
    files = {p for base in ("src", "tests", "scripts") for p in (PROJECT_ROOT / base).rglob("*.py")}
    files.update((PROJECT_ROOT / "src").rglob("*.f90"))
    for relative in (
        "README.md", "requirements.txt", "docs/features_v2.md",
        "docs/decisions/044_h1_objective_execution_repair.md",
        "docs/decisions/045_h1_readout_operational_clarifications.md",
        "docs/decisions/046_h1_e1_dependence_and_claim_scope.md",
        "docs/registrations/h1_e1_panel_v2.json",
        "docs/reports/2026-10-01_h1_repair_audit.md",
        "docs/reports/2026-10-01_h1_launch_readiness.md",
        "docs/reports/2026-10-01_h1_design_repair_audit.md",
        "docs/manuscript/corrected_e1_study_outline.md",
        "data/v2/static_feature_matrix.csv", "data/v2/static_feature_tiers.json",
        "data/v2/corpora_manifest.json",
    ):
        path = PROJECT_ROOT / relative
        if path.exists():
            files.add(path)
    hashes = {}
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED) as archive_file:
        for path in sorted(files):
            relative = path.relative_to(PROJECT_ROOT).as_posix()
            archive_file.write(path, relative)
            hashes[relative] = text_hash(path)
    tests = set()
    for path in root.glob("design_tests*.xml"):
        for case in ET.parse(path).iter("testcase"):
            if not any(case.find(tag) is not None for tag in ("failure", "error", "skipped")):
                tests.add((case.attrib.get("classname"), case.attrib.get("name")))
    obj = H1Objective.load(FREEZE_COMMIT, FEATURE_SHA256, 20000)
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip(),
        "runtime": {"python": sys.version, "numpy": np.__version__, "platform": platform.platform()},
        "protocol_identity": protocol_identity(), "objective_identity": obj.provenance["identity_sha256"],
        "reference_inputs_sha256_lf": reference_input_hashes(), "source_hashes_lf": hashes,
        "snapshot_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        "distinct_passing_tests_this_phase": len(tests),
        "production_launched": False,
        "scope_choice": "pending_author_response",
        "interpretation": "Source snapshot includes preserved pre-existing repository work; raw source corpora remain at their recorded paths, with reference hashes recorded",
    }
    with (root / "completion_inventory.json").open("x", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(json.dumps({k: payload[k] for k in ("snapshot_sha256", "distinct_passing_tests_this_phase", "objective_identity", "production_launched", "scope_choice")}, indent=2))


if __name__ == "__main__":
    archive()
