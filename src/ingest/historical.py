"""
Local Historical Corpus Ingester
================================
Purpose:
    Ingest attested historical text that already exists on disk into the same
    tokenized / fingerprinted format used by the rest of Project RBT.

    This module is intentionally local-file-first. It does not fetch from the
    network. The goal is to let us drop an Old French or other historical
    validator corpus into the repo, normalize it, fingerprint it, and compare
    it against the French bridge checkpoints.

Inputs:
    data/raw/historical/{name}/**/*.txt
    data/raw/historical/{name}/**/*.md

Outputs:
    data/processed/historical/{name}_tokens.json
    data/processed/historical/{name}_manifest.json
    data/matrices/{name}_cooccurrence.npy
    data/matrices/{name}_cooccurrence_meta.json
    data/matrices/{name}_positional.npy
    data/matrices/{name}_positional_meta.json
    data/matrices/{name}_ngram_meta.json

Usage:
    python -m src.ingest.historical --name old_french --language french --period "Old French"
    python -m src.ingest.historical --name old_french --input-dir path/to/texts
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import date
from pathlib import Path

from src.fingerprint import cooccurrence, ngram, positional
from src.ingest.tokenize import corpus_stats, tokenize_corpus

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_HISTORICAL_DIR = PROJECT_ROOT / "data" / "raw" / "historical"
PROCESSED_HISTORICAL_DIR = PROJECT_ROOT / "data" / "processed" / "historical"
MATRICES_DIR = PROJECT_ROOT / "data" / "matrices"

SUPPORTED_TEXT_EXTENSIONS = {".txt", ".md"}
IGNORED_BASENAMES = {"readme.md", "readme.txt", "sources.md", "sources.txt"}


def _discover_text_paths(input_dir: Path) -> list[Path]:
    paths = [
        path
        for path in sorted(input_dir.rglob("*"))
        if (
            path.is_file()
            and path.suffix.lower() in SUPPORTED_TEXT_EXTENSIONS
            and path.name.lower() not in IGNORED_BASENAMES
            and not path.name.startswith("_")
        )
    ]
    return paths


def _read_texts(paths: list[Path]):
    for path in paths:
        yield path.read_text(encoding="utf-8")


def ingest(
    name: str,
    language: str,
    period_label: str,
    input_dir: Path | None = None,
    output_dir: Path = PROCESSED_HISTORICAL_DIR,
    matrices_dir: Path = MATRICES_DIR,
    source: str = "local_text_files",
    notes: str = "",
    build_fingerprints: bool = True,
) -> Path:
    """
    Ingest a local historical corpus from plain-text files.

    Args:
        name: Corpus id used for output file naming, e.g. "old_french".
        language: Modern branch label, e.g. "french".
        period_label: Human-readable stage label, e.g. "Old French".
        input_dir: Optional override for raw input directory.
        output_dir: Directory for tokenized historical corpus JSON.
        matrices_dir: Directory for derived fingerprint files.
        source: Metadata label for provenance.
        notes: Free-form manifest note.
        build_fingerprints: Whether to generate matrices / n-gram metadata.

    Returns:
        Path to the written tokenized corpus JSON.
    """
    if input_dir is None:
        input_dir = RAW_HISTORICAL_DIR / name

    if not input_dir.exists():
        raise FileNotFoundError(f"Historical input directory does not exist: {input_dir}")

    text_paths = _discover_text_paths(input_dir)
    if not text_paths:
        raise ValueError(
            f"No supported text files found in {input_dir} "
            f"(expected {sorted(SUPPORTED_TEXT_EXTENSIONS)})"
        )

    log.info(
        "Ingesting historical corpus '%s' from %d files in %s",
        name,
        len(text_paths),
        input_dir,
    )

    sequences = tokenize_corpus(_read_texts(text_paths), log_interval=25)
    if not sequences:
        raise ValueError(f"Historical corpus '{name}' produced zero token sequences")

    stats = corpus_stats(sequences)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokens_path = output_dir / f"{name}_tokens.json"
    manifest_path = output_dir / f"{name}_manifest.json"

    corpus = {
        "language": name,
        "branch_language": language,
        "historical_period": period_label,
        "source": source,
        "ingest_date": date.today().isoformat(),
        "input_dir": str(input_dir),
        "file_count": len(text_paths),
        "input_files": [str(path) for path in text_paths],
        "total_tokens": stats["total_tokens"],
        "sequence_count": stats["sequence_count"],
        "unique_types": stats["unique_types"],
        "type_token_ratio": stats["type_token_ratio"],
        "mean_seq_length": stats["mean_seq_length"],
        "notes": notes,
        "sequences": sequences,
    }

    with tokens_path.open("w", encoding="utf-8") as fh:
        json.dump(corpus, fh, ensure_ascii=False, indent=2)
    log.info("Wrote historical tokens to %s", tokens_path)

    manifest = {k: v for k, v in corpus.items() if k != "sequences"}
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
    log.info("Wrote historical manifest to %s", manifest_path)

    if build_fingerprints:
        cooccurrence.run_from_sequences(name, sequences, output_dir=matrices_dir)
        positional.run_from_sequences(name, sequences, output_dir=matrices_dir)
        ngram.run_from_sequences(name, sequences, output_dir=matrices_dir)
        log.info("Historical fingerprints written to %s", matrices_dir)

    return tokens_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest a local attested historical corpus")
    parser.add_argument("--name", required=True, help="Corpus id, e.g. old_french")
    parser.add_argument(
        "--language",
        required=True,
        help="Modern branch label, e.g. french",
    )
    parser.add_argument(
        "--period",
        required=True,
        help="Human-readable historical period label, e.g. Old French",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Override input directory (defaults to data/raw/historical/{name})",
    )
    parser.add_argument(
        "--source",
        default="local_text_files",
        help="Manifest provenance label",
    )
    parser.add_argument(
        "--notes",
        default="",
        help="Optional manifest note",
    )
    args = parser.parse_args()

    ingest(
        name=args.name,
        language=args.language,
        period_label=args.period,
        input_dir=args.input_dir,
        source=args.source,
        notes=args.notes,
    )
