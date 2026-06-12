"""
Build the Project RBT v2 corpus provenance manifest.

The output is a registry, not a token dump. It points to existing local corpus
manifests, v2 Leipzig manifests, and planned or blocked rows required by the
v2 pre-registration draft.

Usage:
    python -m src.ingest.v2_corpus_manifest
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.ingest.leipzig import LEIPZIG_PRESETS
from src.ingest.tatoeba import TATOEBA_PRESETS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "v2" / "corpora_manifest.json"
DEFAULT_V2_CORPORA_DIR = PROJECT_ROOT / "data" / "v2" / "corpora"


@dataclass(frozen=True)
class LocalCorpusSpec:
    corpus_id: str
    language: str
    role: str
    family: str
    tokens_path: str
    manifest_path: str | None = None
    required_for_h1: bool = True
    sequestered: bool = False
    notes: str = ""


LOCAL_CORPORA: tuple[LocalCorpusSpec, ...] = (
    LocalCorpusSpec(
        corpus_id="latin",
        language="latin",
        role="target",
        family="italic",
        tokens_path="data/sequestered/latin/latin_tokens.json",
        manifest_path="data/sequestered/latin/latin_manifest.json",
        sequestered=True,
    ),
    LocalCorpusSpec(
        corpus_id="french",
        language="french",
        role="romance_source",
        family="romance",
        tokens_path="data/processed/romance/french_tokens.json",
        manifest_path="data/processed/romance/french_manifest.json",
    ),
    LocalCorpusSpec(
        corpus_id="spanish",
        language="spanish",
        role="romance_source",
        family="romance",
        tokens_path="data/processed/romance/spanish_tokens.json",
        manifest_path="data/processed/romance/spanish_manifest.json",
    ),
    LocalCorpusSpec(
        corpus_id="italian",
        language="italian",
        role="romance_source",
        family="romance",
        tokens_path="data/processed/romance/italian_tokens.json",
        manifest_path="data/processed/romance/italian_manifest.json",
    ),
    LocalCorpusSpec(
        corpus_id="portuguese",
        language="portuguese",
        role="romance_source_positive_control",
        family="romance",
        tokens_path="data/sequestered/portuguese/portuguese_tokens.json",
        manifest_path="data/sequestered/portuguese/portuguese_manifest.json",
        sequestered=True,
        notes="Positive Romance control; not a reconstruction input during v1/v5 runs.",
    ),
    LocalCorpusSpec(
        corpus_id="romanian",
        language="romanian",
        role="romance_source",
        family="romance",
        tokens_path="data/processed/romance/romanian_tokens.json",
        manifest_path="data/processed/romance/romanian_manifest.json",
    ),
    LocalCorpusSpec(
        corpus_id="old_french",
        language="old_french",
        role="historical_validator",
        family="romance",
        tokens_path="data/processed/historical/old_french_tokens.json",
        manifest_path="data/processed/historical/old_french_manifest.json",
    ),
    LocalCorpusSpec(
        corpus_id="middle_french",
        language="middle_french",
        role="historical_validator",
        family="romance",
        tokens_path="data/processed/historical/middle_french_tokens.json",
        manifest_path="data/processed/historical/middle_french_manifest.json",
    ),
    LocalCorpusSpec(
        corpus_id="langue_d_oil",
        language="langue_d_oil",
        role="historical_validator",
        family="romance",
        tokens_path="data/processed/historical/langue_d_oil_tokens.json",
        manifest_path="data/processed/historical/langue_d_oil_manifest.json",
    ),
    LocalCorpusSpec(
        corpus_id="anglo_norman",
        language="anglo_norman",
        role="historical_validator",
        family="romance",
        tokens_path="data/processed/historical/anglo_norman_tokens.json",
        manifest_path="data/processed/historical/anglo_norman_manifest.json",
    ),
    LocalCorpusSpec(
        corpus_id="old_occitan",
        language="old_occitan",
        role="historical_validator",
        family="romance",
        tokens_path="data/processed/historical/old_occitan_tokens.json",
        manifest_path="data/processed/historical/old_occitan_manifest.json",
    ),
    LocalCorpusSpec(
        corpus_id="old_spanish",
        language="old_spanish",
        role="historical_validator",
        family="romance",
        tokens_path="data/processed/historical/old_spanish_tokens.json",
        manifest_path="data/processed/historical/old_spanish_manifest.json",
    ),
    LocalCorpusSpec(
        corpus_id="sumerian",
        language="sumerian",
        role="typological_control",
        family="language_isolate",
        tokens_path="data/processed/nulls/sumerian/sumerian_tokens.json",
        manifest_path="data/processed/nulls/sumerian/sumerian_manifest.json",
        notes="Structured non-IE ceiling; manifest documents lexical-text limitation.",
    ),
    LocalCorpusSpec(
        corpus_id="markov_noise",
        language="markov_null",
        role="null_control",
        family="generated",
        tokens_path="data/processed/nulls/markov/markov_tokens.json",
        manifest_path=None,
        notes="Generated null corpus; tokens JSON carries its own metadata.",
    ),
    LocalCorpusSpec(
        corpus_id="occitan",
        language="occitan",
        role="auxiliary_modern_romance_positive_diagnostic",
        family="romance",
        tokens_path="data/processed/romance/occitan_tokens.json",
        manifest_path="data/processed/romance/occitan_manifest.json",
        required_for_h1=False,
        notes=(
            "Modern Occitan positive-control diagnostic; visible for H1 reporting "
            "but not required for freeze or static in-loop tier selection."
        ),
    ),
    LocalCorpusSpec(
        corpus_id="genoese",
        language="genoese",
        role="auxiliary_modern_romance",
        family="romance",
        tokens_path="data/processed/romance/genoese_tokens.json",
        manifest_path="data/processed/romance/genoese_manifest.json",
        required_for_h1=False,
    ),
)

REQUIRED_LEIPZIG_IDS = {
    "turkish_news_2024_100k",
    "finnish_news_2022_100k",
    "vietnamese_news_2022_100k",
    "catalan_news_2022_100k",
    "arabic_news_2022_100k",
    "bulgarian_news_2022_100k",
    "german_news_2025_100k",
}

OPTIONAL_PLACEHOLDER_CORPORA: tuple[dict[str, Any], ...] = ()

STAT_KEYS = (
    "sequence_count",
    "total_tokens",
    "unique_types",
    "type_token_ratio",
    "mean_seq_length",
    "pre_cleaning_sentence_count",
    "pre_cleaning_whitespace_token_count",
)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _strip_sequences(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key != "sequences"}


def _path_status(path: Path) -> str:
    return "present" if path.exists() else "missing"


def _metadata_from_manifest_or_small_tokens(
    project_root: Path,
    manifest_rel: str | None,
    tokens_rel: str,
) -> dict[str, Any]:
    if manifest_rel:
        manifest_path = project_root / manifest_rel
        if manifest_path.exists():
            return _read_json(manifest_path)

    tokens_path = project_root / tokens_rel
    if tokens_path.exists() and tokens_path.stat().st_size < 20 * 1024 * 1024:
        return _strip_sequences(_read_json(tokens_path))
    return {}


def _local_entry(spec: LocalCorpusSpec, project_root: Path) -> dict[str, Any]:
    tokens_path = project_root / spec.tokens_path
    manifest_path = project_root / spec.manifest_path if spec.manifest_path else None
    metadata = _metadata_from_manifest_or_small_tokens(
        project_root,
        spec.manifest_path,
        spec.tokens_path,
    )
    status = "ready" if tokens_path.exists() and (manifest_path is None or manifest_path.exists()) else "missing"
    entry = {
        **asdict(spec),
        "status": status,
        "tokens_status": _path_status(tokens_path),
        "manifest_status": _path_status(manifest_path) if manifest_path else "embedded_in_tokens_or_absent",
        "stats": {key: metadata[key] for key in STAT_KEYS if key in metadata},
        "source_metadata": {
            key: value
            for key, value in metadata.items()
            if key
            not in {
                "language",
                "total_tokens",
                "sequence_count",
                "unique_types",
                "type_token_ratio",
                "mean_seq_length",
            }
        },
    }
    return entry


def _leipzig_entry(corpus_id: str, project_root: Path, corpora_dir: Path) -> dict[str, Any]:
    preset = LEIPZIG_PRESETS[corpus_id]
    tokens_rel = _relative_to_project(corpora_dir / f"{corpus_id}_tokens.json", project_root)
    manifest_path = corpora_dir / f"{corpus_id}_manifest.json"
    manifest_rel = _relative_to_project(manifest_path, project_root)
    metadata = _read_json(manifest_path) if manifest_path.exists() else {}

    if manifest_path.exists():
        status = "ready" if preset.tokenization_status == "ready" else "ready_needs_review"
    elif preset.tokenization_status != "ready":
        status = "needs_tokenization_policy"
    else:
        status = "planned_missing"

    entry = {
        "corpus_id": corpus_id,
        "language": preset.language,
        "role": preset.role,
        "family": preset.family,
        "tokens_path": tokens_rel,
        "manifest_path": manifest_rel,
        "required_for_h1": preset.required_for_h1,
        "status": status,
        "tokens_status": _path_status(project_root / tokens_rel),
        "manifest_status": _path_status(manifest_path),
        "source": "leipzig_corpora_collection",
        "source_url": preset.source_url,
        "download_page": preset.download_page,
        "source_type": preset.source_type,
        "source_year": preset.source_year,
        "sentence_count_label": preset.sentence_count_label,
        "tokenization_status": preset.tokenization_status,
        "notes": preset.notes,
        "stats": {key: metadata[key] for key in STAT_KEYS if key in metadata},
    }
    if metadata:
        entry["source_metadata"] = {
            key: value
            for key, value in metadata.items()
            if key
            not in {
                "corpus_id",
                "language",
                "family",
                "role",
                "sequence_count",
                "total_tokens",
                "unique_types",
                "type_token_ratio",
                "mean_seq_length",
            }
        }
    return entry


def _tatoeba_entry(corpus_id: str, project_root: Path, corpora_dir: Path) -> dict[str, Any]:
    preset = TATOEBA_PRESETS[corpus_id]
    tokens_rel = _relative_to_project(corpora_dir / f"{corpus_id}_tokens.json", project_root)
    manifest_path = corpora_dir / f"{corpus_id}_manifest.json"
    manifest_rel = _relative_to_project(manifest_path, project_root)
    metadata = _read_json(manifest_path) if manifest_path.exists() else {}
    status = "ready" if manifest_path.exists() else "planned_missing"

    entry = {
        "corpus_id": corpus_id,
        "language": preset.language,
        "role": preset.role,
        "family": preset.family,
        "tokens_path": tokens_rel,
        "manifest_path": manifest_rel,
        "required_for_h1": preset.required_for_h1,
        "status": status,
        "tokens_status": _path_status(project_root / tokens_rel),
        "manifest_status": _path_status(manifest_path),
        "source": "tatoeba",
        "source_url": preset.source_url,
        "download_page": "https://tatoeba.org/en/downloads",
        "tokenization_status": "ready",
        "tokenization_mode": preset.tokenization_mode,
        "notes": preset.notes,
        "stats": {key: metadata[key] for key in STAT_KEYS if key in metadata},
    }
    if metadata:
        entry["source_metadata"] = {
            key: value
            for key, value in metadata.items()
            if key
            not in {
                "corpus_id",
                "language",
                "family",
                "role",
                "sequence_count",
                "total_tokens",
                "unique_types",
                "type_token_ratio",
                "mean_seq_length",
            }
        }
    return entry


def _relative_to_project(path: Path, project_root: Path) -> str:
    try:
        return str(path.relative_to(project_root)).replace("\\", "/")
    except ValueError:
        return str(path)


def _h1_blockers(entries: list[dict[str, Any]]) -> list[str]:
    blockers: list[str] = []
    for entry in entries:
        if not entry.get("required_for_h1", False):
            continue
        if entry["status"] in {"ready", "ready_needs_review"}:
            if entry["status"] == "ready_needs_review":
                blockers.append(f"{entry['corpus_id']}: tokenization review still required")
            continue
        blockers.append(f"{entry['corpus_id']}: {entry['status']}")
    return blockers


def build_manifest(
    project_root: Path = PROJECT_ROOT,
    corpora_dir: Path | None = None,
) -> dict[str, Any]:
    corpora_dir = corpora_dir or (project_root / "data" / "v2" / "corpora")
    entries = [_local_entry(spec, project_root) for spec in LOCAL_CORPORA]
    leipzig_ids = sorted(REQUIRED_LEIPZIG_IDS | set(LEIPZIG_PRESETS))
    entries.extend(_leipzig_entry(corpus_id, project_root, corpora_dir) for corpus_id in leipzig_ids)
    entries.extend(_tatoeba_entry(corpus_id, project_root, corpora_dir) for corpus_id in sorted(TATOEBA_PRESETS))
    entries.extend(OPTIONAL_PLACEHOLDER_CORPORA)
    entries = sorted(entries, key=lambda entry: (not entry.get("required_for_h1", False), entry["role"], entry["corpus_id"]))

    ready_count = sum(1 for entry in entries if entry["status"] == "ready")
    blockers = _h1_blockers(entries)
    return {
        "schema": "project_rbt_v2_corpus_manifest_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "h1_ready" if not blockers else "h1_blocked",
        "corpus_count": len(entries),
        "ready_count": ready_count,
        "h1_blockers": blockers,
        "corpora": entries,
    }


def write_manifest(
    output_path: Path = DEFAULT_OUTPUT_PATH,
    project_root: Path = PROJECT_ROOT,
    corpora_dir: Path | None = None,
) -> Path:
    manifest = build_manifest(project_root=project_root, corpora_dir=corpora_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
    return output_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the v2 corpus manifest")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--corpora-dir", type=Path, default=DEFAULT_V2_CORPORA_DIR)
    args = parser.parse_args(argv)
    path = write_manifest(output_path=args.output, corpora_dir=args.corpora_dir)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
