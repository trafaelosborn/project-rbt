"""Frozen H1 measurement, objective, and execution provenance.

See Decision 044 for the disclosed sampling boundary policy. Keep legacy
four-component diagnostics separate from this six-component reward vector.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FREEZE_COMMIT = "232393a2d943f878b71d9c75aa40dce8d4602e4f"
FEATURE_SHA256 = "551a94d99b5426bf15968b9edf1769b76e609e6fe2927a1c8e2fdf30b9990a87"
OBJECTIVE_ID = "h1_frozen_6d_v1"
FROZEN_HASHES = {
    "docs/features_v2.md": FEATURE_SHA256,
    "data/v2/static_feature_matrix.csv": "fc443ede9665e36690c84be0fbb433c50e668bbcea2198e1c2c268f0774a0b5e",
    "data/v2/static_feature_tiers.json": "aa2b2afe6b8daa65bada3c827c6bc05b1ca9189126299da931b642fe69682404",
    "data/v2/corpora_manifest.json": "c0b4eb1174d26777a6b4653e3a11621338c4fa23ad65911739439714c8df2696",
}
FEATURES = (
    "f02_top100_bigram_coverage", "f09_word_length_mean", "f15_suffix_entropy",
    "f17_conditional_char_entropy", "f18_dislegomena_to_hapax",
    "f19_word_edge_entropy_asymmetry",
)
IMPLEMENTATION_FILES = (
    "src/retrodiction/h1_objective.py", "src/validation/static_feature_matrix.py",
    "src/retrodiction/engine_reinforced_v2.py", "src/retrodiction/engine_reinforced_v3.py",
    "src/retrodiction/engine_reinforced_v4.py", "src/retrodiction/engine_reinforced_v5.py",
    "src/retrodiction/long_run_v4.py", "src/retrodiction/long_run_v5.py",
    "src/retrodiction/h1_execution.py",
    "src/retrodiction/engine_reinforced.py", "src/retrodiction/similarity.py",
    "src/fingerprint/ngram.py", "src/ingest/tokenize.py",
    "src/validation/hungarian_alignment.py",
)


def text_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def stable_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


class BudgetError(ValueError):
    """A candidate cannot supply the common measurement budget."""


def measurement_prefix(sequences: list[list[str]], budget: int) -> list[list[str]]:
    """A deterministic measurement window; never shuffle or pad a candidate."""
    selected = []
    remaining = budget
    for seq in sequences:
        if seq:
            selected.append(list(seq[:remaining]))
            remaining -= len(selected[-1])
        if remaining == 0:
            return selected
    raise BudgetError(f"Corpus has {budget - remaining} tokens, below H1 budget {budget}")


class H1Objective:
    def __init__(self, rows: list[dict], features: tuple[str, ...], budget: int,
                 provenance: dict, score_scale: float = 5.0):
        self.rows = rows
        self.features = features
        self.budget = budget
        self.score_scale = score_scale
        self.columns = tuple(key for key in rows[0] if key.startswith("f") and key[1:3].isdigit())
        matrix = np.array([[float(row[c]) for c in self.columns] for row in rows])
        self.means = matrix.mean(axis=0)
        self.scales = matrix.std(axis=0, ddof=0)
        if not np.isfinite(matrix).all() or np.any(self.scales <= 0):
            raise ValueError("H1 calibration contains nonfinite or constant features")
        self.indices = [self.columns.index(c) for c in features]
        latin = [row for row in rows if row["language"] == "latin"]
        if len(latin) != 1:
            raise ValueError("H1 requires exactly one frozen Latin calibration row")
        self.latin_raw = np.array([float(latin[0][c]) for c in self.columns])
        self.latin_z = (self.latin_raw - self.means) / self.scales
        self.provenance = {
            **provenance, "objective_id": OBJECTIVE_ID, "features": list(features),
            "corpus_budget": budget, "score_scale": score_scale,
            "scaling_ddof": 0, "reference_count": len(rows),
            "columns_21d": list(self.columns), "means_21d": self.means.tolist(),
            "scales_21d": self.scales.tolist(), "latin_raw_21d": self.latin_raw.tolist(),
            "sampling_policy": "seeded_initial_shuffle_then_fixed_order_prefix_v1",
            "undersized_candidate_policy": "reject_without_padding",
            "effective_candidate_scoring_backend": "h1_python_reference",
        }
        self.provenance["identity_sha256"] = stable_hash(self.provenance)

    @classmethod
    def load(cls, commit: str, feature_hash: str, budget: int,
             root: Path = PROJECT_ROOT) -> "H1Objective":
        if commit != FREEZE_COMMIT or feature_hash != FEATURE_SHA256:
            raise ValueError("H1 freeze identifiers do not match the approved registration")
        for relative, expected in FROZEN_HASHES.items():
            if text_hash(root / relative) != expected:
                raise ValueError(f"Frozen H1 artifact changed: {relative}")
        tiers = json.loads((root / "data/v2/static_feature_tiers.json").read_text(encoding="utf-8"))
        if tuple(tiers["retained_in_loop"]) != FEATURES:
            raise ValueError("H1 retained feature order differs from the approved tier")
        rows = list(csv.DictReader(io.StringIO(
            (root / "data/v2/static_feature_matrix.csv").read_text(encoding="utf-8"))))
        if budget != 20_000 or len(rows) != 22 or any(int(r["corpus_budget"]) != budget for r in rows):
            raise ValueError("H1 currently requires the frozen 22-row, 20000-token calibration")
        code_hashes = {p: text_hash(root / p) for p in IMPLEMENTATION_FILES}
        return cls(rows, FEATURES, budget, {
            "features_freeze_commit": commit, "features_freeze_sha256": feature_hash,
            "frozen_artifact_hashes_lf": FROZEN_HASHES,
            "implementation_hashes_lf": code_hashes,
        })

    @classmethod
    def from_config(cls, config):
        commit = getattr(config, "features_freeze_commit", "")
        digest = getattr(config, "features_freeze_sha256", "")
        if not commit and not digest:
            return None
        if getattr(config, "enable_culture_bombs", False) or getattr(config, "use_semantic_transparency", False):
            raise ValueError("Frozen H1 does not enable culture bombs or semantic transparency")
        return cls.load(commit, digest, config.corpus_budget)

    def measure(self, sequences):
        return measurement_prefix(sequences, self.budget)

    def evaluate(self, sequences) -> dict:
        # Lazy import avoids the existing static-feature -> engine-v2 import cycle.
        from src.validation.static_feature_matrix import compute_feature_vector
        measured = self.measure(sequences)
        values = compute_feature_vector(measured)
        raw = np.array([values[c] for c in self.columns])
        z = (raw - self.means) / self.scales
        if not np.isfinite(z).all():
            raise ValueError("Nonfinite H1 candidate features")
        delta = z[self.indices] - self.latin_z[self.indices]
        return {
            "objective_id": OBJECTIVE_ID,
            "identity_sha256": self.provenance["identity_sha256"],
            "measured_token_count": self.budget,
            "state_token_count": sum(map(len, sequences)),
            "features": values, "z_6d": z[self.indices].tolist(),
            "latin_delta_6d": delta.tolist(),
            "structural_score": float(-self.score_scale * np.linalg.norm(delta)),
        }

    def reference_distances(self, sequences) -> list[dict]:
        measurement = self.evaluate(sequences)
        z = (np.array([measurement["features"][c] for c in self.columns]) - self.means) / self.scales
        return [{
            "corpus_id": r["corpus_id"], "language": r["language"], "role": r["role"],
            "euclidean_6d": float(np.linalg.norm((z - (np.array([float(r[c]) for c in self.columns]) - self.means) / self.scales)[self.indices])),
            "euclidean_21d": float(np.linalg.norm(z - (np.array([float(r[c]) for c in self.columns]) - self.means) / self.scales)),
        } for r in self.rows]
