"""Pinned post-freeze E1 protocol and pre-launch registration sidecar."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from src.retrodiction.h1_objective import (
    PROJECT_ROOT, H1Objective, FREEZE_COMMIT, FEATURE_SHA256, stable_hash, text_hash,
)

PROTOCOL_PATH = "docs/registrations/h1_e1_panel_v2.json"
PROTOCOL_SHA256 = "cc6dcab72d37296efea765765b32c53560abbe730fee7923ca69fe38015f6a7a"
DECISION_PATH = "docs/decisions/046_h1_e1_dependence_and_claim_scope.md"
DECISION_SHA256 = "db81921468c807d52539642a24fa503ae496effc0ab5762f50e9fadbe9e3ab89"
ANALYSIS_FILES = ("src/validation/h1_convergence.py", "src/validation/h1_protocol.py")
SIDECAR = "study_registration.json"


def protocol_identity():
    for path, expected in ((PROTOCOL_PATH, PROTOCOL_SHA256), (DECISION_PATH, DECISION_SHA256)):
        if text_hash(PROJECT_ROOT / path) != expected:
            raise ValueError(f"Pinned E1 amendment changed: {path}")
    return {"protocol": json.loads((PROJECT_ROOT / PROTOCOL_PATH).read_text(encoding="utf-8")),
            "protocol_sha256_lf": PROTOCOL_SHA256, "decision_sha256_lf": DECISION_SHA256,
            "analysis_files_sha256_lf": {p: text_hash(PROJECT_ROOT / p) for p in ANALYSIS_FILES}}


def job_config(job):
    from src.retrodiction.long_run_v4 import LongRunConfig
    fields = job.to_dict()
    fields["n_candidates"] = fields.pop("candidate_count")
    fields["start_corpus"] = job.start_corpus
    fields["output_dir"] = job.output_dir
    return LongRunConfig(**fields).to_dict()


def reference_input_hashes():
    """Also pin the retained form/coherence/family reference inputs."""
    from src.retrodiction.similarity import REAL_LANGUAGE_REFERENCE_LANGS, PROCESSED_DIR, MATRICES_DIR
    files = {PROJECT_ROOT / "data/sequestered/latin/latin_tokens.json",
             MATRICES_DIR / "latin_ngram_meta.json"}
    for language in set(REAL_LANGUAGE_REFERENCE_LANGS) | {"markov", "sumerian"}:
        path = (PROCESSED_DIR / "nulls" / language / f"{language}_tokens.json"
                if language in {"markov", "sumerian"}
                else PROCESSED_DIR / "romance" / f"{language}_tokens.json")
        meta = path.parent.parent.parent / "matrices" / f"{language}_ngram_meta.json"
        if not meta.exists():
            meta = MATRICES_DIR / f"{language}_ngram_meta.json"
        files.update((path, meta))
    return {str(p.relative_to(PROJECT_ROOT)).replace("\\", "/"): text_hash(p) for p in sorted(files)}


def write_registration(root: Path, jobs):
    """Seal an entirely new production directory; never adopt existing runs."""
    from src.control.h1_matrix import build_jobs
    root = root.resolve()
    if root.exists():
        raise ValueError("Production registration requires a new output root")
    expected = [j.to_dict() for j in build_jobs(root)]
    if [j.to_dict() for j in jobs] != expected:
        raise ValueError("Jobs differ from the registered production panel/settings")
    identity = protocol_identity()
    obj = H1Objective.load(FREEZE_COMMIT, FEATURE_SHA256, 20000)
    payload = {
        "schema": "h1_study_registration_v2", "created_utc": datetime.now(timezone.utc).isoformat(),
        "output_root": str(root), **identity, "objective_identity": obj.provenance["identity_sha256"],
        "reference_inputs_sha256_lf": reference_input_hashes(),
        "jobs": [{"job_id": j.job_id, "config": job_config(j), "command": j.command(),
                  "source_sha256_lf": text_hash(j.start_corpus)} for j in jobs],
    }
    payload["registration_sha256"] = stable_hash(payload)
    root.mkdir(parents=True)
    (root / SIDECAR).write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    return payload


def validate_registration(root: Path):
    root = root.resolve()
    path = root / SIDECAR
    if not path.exists():
        raise ValueError("Missing pre-launch study registration; existing runs cannot be retroactively registered")
    payload = json.loads(path.read_text(encoding="utf-8"))
    unsigned = {k: v for k, v in payload.items() if k != "registration_sha256"}
    if payload.get("registration_sha256") != stable_hash(unsigned):
        raise ValueError("Study registration integrity failure")
    if payload.get("schema") != "h1_study_registration_v2" or payload["output_root"] != str(root):
        raise ValueError("Study registration schema/output root mismatch")
    if any(payload.get(k) != v for k, v in protocol_identity().items()):
        raise ValueError("E1 protocol or analysis implementation changed since registration")
    obj = H1Objective.load(FREEZE_COMMIT, FEATURE_SHA256, 20000)
    if payload["objective_identity"] != obj.provenance["identity_sha256"]:
        raise ValueError("Objective changed since study registration")
    if payload.get("reference_inputs_sha256_lf") != reference_input_hashes():
        raise ValueError("Form/coherence/family reference inputs changed since registration")
    from src.control.h1_matrix import build_jobs
    expected = build_jobs(root)
    if [j["config"] for j in payload["jobs"]] != [job_config(j) for j in expected]:
        raise ValueError("Study registration job contracts differ from the protocol")
    for registered, job in zip(payload["jobs"], expected):
        if (registered["job_id"] != job.job_id or registered["command"] != job.command()
                or registered["source_sha256_lf"] != text_hash(job.start_corpus)):
            raise ValueError("Registered source/command changed")
    return payload


def validate_registered_manifest(registration, manifest):
    cfg = manifest["config"]
    identifier = f"{cfg['language']}_seed_{cfg['seed']}"
    matches = [j for j in registration["jobs"] if j["job_id"] == identifier]
    if len(matches) != 1 or cfg != matches[0]["config"]:
        raise ValueError(f"Run differs from the pre-launch job contract: {identifier}")
    if datetime.fromisoformat(manifest["created_utc"]) < datetime.fromisoformat(registration["created_utc"]):
        raise ValueError(f"Run predates study registration: {identifier}")
    execution = manifest["h1_execution"]
    if (execution["source_sha256_lf"] != matches[0]["source_sha256_lf"]
            or execution["objective"]["identity_sha256"] != registration["objective_identity"]):
        raise ValueError(f"Run source/objective differs from study registration: {identifier}")
