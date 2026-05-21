"""
Statistical Fingerprint Container
==================================
Purpose:
    Container class for a complete statistical fingerprint of a language corpus
    or retrodiction stage. Holds references to the four fingerprint components
    (co-occurrence matrix, positional distribution, n-gram profiles, type/token ratio)
    and handles serialization to/from the bridge stage record format.

Bridge stage record format (from spec):
    {
      "stage_id": "FR_retro_014",
      "source_language": "french",
      "iteration": 14,
      "fingerprint": {
        "cooccurrence_matrix": "data/matrices/FR_retro_014_cooc.npy",
        "positional_dist":     "data/matrices/FR_retro_014_pos.npy",
        "type_token_ratio":    0.412,
        "bigram_profile":      "data/matrices/FR_retro_014_bg.json",
        "trigram_profile":     "data/matrices/FR_retro_014_tg.json"
      },
      "scores": {
        "vs_markov_noise":         0.891,
        "vs_sumerian":             0.743,
        "vs_portuguese_control":   0.612,
        "vs_latin_ground_truth":   null
      },
      "notes": "",
      "flags": []
    }

The vs_latin_ground_truth field remains null until the sequestration firewall is
lifted. It is populated in a separate post-validation pass.

Usage:
    fp = StatisticalFingerprint.from_language("french")
    fp.scores["vs_markov_noise"] = 0.891
    fp.save("results/bridges/FR_retro_014.json")

    fp2 = StatisticalFingerprint.load("results/bridges/FR_retro_014.json")
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MATRICES_DIR = PROJECT_ROOT / "data" / "matrices"


@dataclass
class StatisticalFingerprint:
    """
    A complete statistical fingerprint for one language corpus or retrodiction stage.

    Matrix fields hold numpy arrays when loaded into memory; the path fields hold
    the corresponding file paths for serialization. On save, arrays are written to
    .npy files and the record stores paths. On load, paths are recorded but arrays
    are not automatically loaded (call load_arrays() explicitly to avoid memory issues
    when iterating over many bridge stages).
    """

    stage_id: str
    source_language: str
    iteration: int

    # File paths (always set)
    cooccurrence_path: Path | None = None
    positional_path: Path | None = None
    bigram_path: Path | None = None
    trigram_path: Path | None = None

    # In-memory arrays (populated by load_arrays() or set directly)
    cooccurrence_matrix: np.ndarray | None = None
    positional_matrix: np.ndarray | None = None

    # Scalar / dict components (always in memory)
    type_token_ratio: float = 0.0
    bigram_profile: dict[str, float] = field(default_factory=dict)
    trigram_profile: dict[str, float] = field(default_factory=dict)

    # Scores against reference corpora
    scores: dict[str, float | None] = field(default_factory=lambda: {
        "vs_markov_noise": None,
        "vs_sumerian": None,
        "vs_portuguese_control": None,
        "vs_latin_ground_truth": None,
    })

    notes: str = ""
    flags: list[str] = field(default_factory=list)

    # ---------------------------------------------------------------------------
    # Array I/O
    # ---------------------------------------------------------------------------

    def load_arrays(self) -> None:
        """Load co-occurrence and positional matrices from disk into memory."""
        if self.cooccurrence_path and self.cooccurrence_path.exists():
            self.cooccurrence_matrix = np.load(self.cooccurrence_path)
            log.info("Loaded co-occurrence matrix from %s", self.cooccurrence_path)
        if self.positional_path and self.positional_path.exists():
            self.positional_matrix = np.load(self.positional_path)
            log.info("Loaded positional matrix from %s", self.positional_path)

    def save_arrays(self, output_dir: Path = MATRICES_DIR) -> None:
        """Write in-memory arrays to .npy files under output_dir."""
        output_dir.mkdir(parents=True, exist_ok=True)
        if self.cooccurrence_matrix is not None:
            path = output_dir / f"{self.stage_id}_cooc.npy"
            np.save(path, self.cooccurrence_matrix)
            self.cooccurrence_path = path
            log.info("Saved co-occurrence matrix to %s", path)
        if self.positional_matrix is not None:
            path = output_dir / f"{self.stage_id}_pos.npy"
            np.save(path, self.positional_matrix)
            self.positional_path = path
            log.info("Saved positional matrix to %s", path)
        if self.bigram_profile:
            path = output_dir / f"{self.stage_id}_bg.json"
            with path.open("w", encoding="utf-8") as fh:
                json.dump(self.bigram_profile, fh, ensure_ascii=False, indent=2)
            self.bigram_path = path
        if self.trigram_profile:
            path = output_dir / f"{self.stage_id}_tg.json"
            with path.open("w", encoding="utf-8") as fh:
                json.dump(self.trigram_profile, fh, ensure_ascii=False, indent=2)
            self.trigram_path = path

    # ---------------------------------------------------------------------------
    # Serialization
    # ---------------------------------------------------------------------------

    def to_record(self) -> dict[str, Any]:
        """Serialize to the bridge stage record format."""
        return {
            "stage_id": self.stage_id,
            "source_language": self.source_language,
            "iteration": self.iteration,
            "fingerprint": {
                "cooccurrence_matrix": str(self.cooccurrence_path) if self.cooccurrence_path else None,
                "positional_dist": str(self.positional_path) if self.positional_path else None,
                "type_token_ratio": self.type_token_ratio,
                "bigram_profile": str(self.bigram_path) if self.bigram_path else None,
                "trigram_profile": str(self.trigram_path) if self.trigram_path else None,
            },
            "scores": self.scores,
            "notes": self.notes,
            "flags": self.flags,
        }

    def save(self, record_path: Path, array_dir: Path = MATRICES_DIR) -> None:
        """Save arrays to disk then write the stage record JSON."""
        self.save_arrays(array_dir)
        record_path.parent.mkdir(parents=True, exist_ok=True)
        with record_path.open("w", encoding="utf-8") as fh:
            json.dump(self.to_record(), fh, ensure_ascii=False, indent=2)
        log.info("Saved stage record to %s", record_path)

    @classmethod
    def load(cls, record_path: Path) -> "StatisticalFingerprint":
        """Load a stage record from JSON. Arrays are NOT automatically loaded into memory."""
        with record_path.open(encoding="utf-8") as fh:
            rec = json.load(fh)

        fp_data = rec.get("fingerprint", {})
        return cls(
            stage_id=rec["stage_id"],
            source_language=rec["source_language"],
            iteration=rec["iteration"],
            cooccurrence_path=Path(fp_data["cooccurrence_matrix"]) if fp_data.get("cooccurrence_matrix") else None,
            positional_path=Path(fp_data["positional_dist"]) if fp_data.get("positional_dist") else None,
            type_token_ratio=fp_data.get("type_token_ratio", 0.0),
            bigram_path=Path(fp_data["bigram_profile"]) if fp_data.get("bigram_profile") else None,
            trigram_path=Path(fp_data["trigram_profile"]) if fp_data.get("trigram_profile") else None,
            scores=rec.get("scores", {}),
            notes=rec.get("notes", ""),
            flags=rec.get("flags", []),
        )

    # ---------------------------------------------------------------------------
    # Factory
    # ---------------------------------------------------------------------------

    @classmethod
    def from_language(
        cls,
        language: str,
        iteration: int = 0,
        matrix_dir: Path = MATRICES_DIR,
    ) -> "StatisticalFingerprint":
        """
        Build a fingerprint for a modern language corpus (iteration=0) by loading
        pre-built matrix files from the standard locations.
        """
        stage_id = f"{language[:2].upper()}_retro_{iteration:03d}"
        fp = cls(
            stage_id=stage_id,
            source_language=language,
            iteration=iteration,
            cooccurrence_path=matrix_dir / f"{language}_cooccurrence.npy",
            positional_path=matrix_dir / f"{language}_positional.npy",
            bigram_path=matrix_dir / f"{language}_ngram_meta.json",
            trigram_path=matrix_dir / f"{language}_ngram_meta.json",
        )

        # Load n-gram profiles and TTR from the combined ngram meta file.
        ngram_meta_path = matrix_dir / f"{language}_ngram_meta.json"
        if ngram_meta_path.exists():
            with ngram_meta_path.open(encoding="utf-8") as fh:
                ngram_meta = json.load(fh)
            fp.bigram_profile = ngram_meta.get("bigrams", {})
            fp.trigram_profile = ngram_meta.get("trigrams", {})
            fp.type_token_ratio = ngram_meta.get("type_token_ratio", 0.0)

        return fp
