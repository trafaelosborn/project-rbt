"""
Tatoeba ingester for v2 optional control corpora.

The immediate use case is Klingon. Tatoeba publishes per-language sentence
exports, and Klingon is small enough to download directly without touching the
large all-language export.

Default outputs:
    data/v2/raw/tatoeba/{archive}.bz2
    data/v2/corpora/{corpus_id}_tokens.json
    data/v2/corpora/{corpus_id}_manifest.json

Usage:
    python -m src.ingest.tatoeba list
    python -m src.ingest.tatoeba ingest --corpus-id klingon_tatoeba_tlh
"""

from __future__ import annotations

import argparse
import bz2
import hashlib
import json
import logging
import re
import unicodedata
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from urllib.parse import urlparse

import requests

from src.ingest.tokenize import corpus_stats, tokenize_sentence

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
V2_DIR = PROJECT_ROOT / "data" / "v2"
DEFAULT_RAW_DIR = V2_DIR / "raw" / "tatoeba"
DEFAULT_OUTPUT_DIR = V2_DIR / "corpora"

TATOEBA_DOWNLOADS_URL = "https://tatoeba.org/en/downloads"
TATOEBA_LICENSE_NOTE = "Tatoeba sentence exports are listed as CC BY 2.0 FR."
USER_AGENT = (
    "ProjectRBT/2.0 (v2 Tatoeba corpus ingestion; "
    "https://github.com/spaceranger-press/project-rbt; contact via GitHub)"
)

_KLINGON_WORD_RE = re.compile(r"[A-Za-z']*[A-Za-z][A-Za-z']*")


@dataclass(frozen=True)
class TatoebaCorpusPreset:
    corpus_id: str
    language: str
    iso_code: str
    family: str
    role: str
    source_url: str
    tokenization_mode: str
    required_for_h1: bool = False
    default_ingest: bool = True
    notes: str = ""


TATOEBA_PRESETS: dict[str, TatoebaCorpusPreset] = {
    "klingon_tatoeba_tlh": TatoebaCorpusPreset(
        corpus_id="klingon_tatoeba_tlh",
        language="klingon",
        iso_code="tlh",
        family="constructed",
        role="typological_control_optional",
        source_url="https://downloads.tatoeba.org/exports/per_language/tlh/tlh_sentences.tsv.bz2",
        tokenization_mode="klingon_latin_case_sensitive",
        notes=(
            "Optional constructed-language control. Klingon Latin orthography "
            "uses uppercase letters and apostrophe phonemically, so this corpus "
            "uses a case-sensitive tokenizer instead of the default lowercasing tokenizer."
        ),
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def archive_name_from_url(url: str) -> str:
    name = Path(urlparse(url).path).name
    if not name.endswith(".bz2"):
        raise ValueError(f"Expected a .bz2 URL, got: {url}")
    return name


def download_archive(
    url: str,
    raw_dir: Path = DEFAULT_RAW_DIR,
    force: bool = False,
) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    archive_path = raw_dir / archive_name_from_url(url)
    if archive_path.exists() and not force:
        log.info("Using cached Tatoeba archive: %s", archive_path)
        return archive_path

    tmp_path = archive_path.with_name(f"{archive_path.name}.tmp")
    log.info("Downloading %s", url)
    with requests.get(url, stream=True, timeout=120, headers={"User-Agent": USER_AGENT}) as resp:
        resp.raise_for_status()
        with tmp_path.open("wb") as fh:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    fh.write(chunk)
    tmp_path.replace(archive_path)
    log.info("Cached Tatoeba archive: %s", archive_path)
    return archive_path


def tokenize_klingon_latin(sentence: str) -> list[str]:
    """
    Tokenize standard Latin-script Klingon without lowercasing.

    Klingon orthography treats several uppercase letters and the apostrophe as
    contrastive letters, so the default Project RBT tokenizer would collapse
    signal that should be retained for this optional constructed-language row.
    """
    sentence = unicodedata.normalize("NFC", sentence).replace("’", "'")
    return _KLINGON_WORD_RE.findall(sentence)


def tokenize_for_mode(sentence: str, mode: str) -> list[str]:
    if mode == "klingon_latin_case_sensitive":
        return tokenize_klingon_latin(sentence)
    if mode == "unicode_word":
        return tokenize_sentence(sentence)
    raise ValueError(f"Unknown Tatoeba tokenization mode: {mode}")


def sentence_from_tatoeba_line(line: str, expected_lang: str) -> tuple[str, str]:
    parts = line.rstrip("\n\r").split("\t", 2)
    if len(parts) != 3:
        raise ValueError(f"Malformed Tatoeba sentence row: {line[:80]!r}")
    _, lang, text = parts
    if lang != expected_lang:
        raise ValueError(f"Expected Tatoeba language {expected_lang!r}, got {lang!r}")
    return lang, text.strip()


def extract_token_sequences(
    archive_path: Path,
    *,
    expected_lang: str,
    tokenization_mode: str,
    max_sentences: int | None = None,
) -> tuple[list[list[str]], dict]:
    sequences: list[list[str]] = []
    raw_sentence_count = 0
    skipped_empty_after_tokenization = 0
    pre_cleaning_whitespace_token_count = 0
    pre_cleaning_character_count = 0

    with bz2.open(archive_path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            _, sentence = sentence_from_tatoeba_line(line, expected_lang)
            if not sentence:
                continue
            raw_sentence_count += 1
            pre_cleaning_whitespace_token_count += len(sentence.split())
            pre_cleaning_character_count += len(sentence)
            tokens = tokenize_for_mode(sentence, tokenization_mode)
            if tokens:
                sequences.append(tokens)
            else:
                skipped_empty_after_tokenization += 1
            if max_sentences is not None and raw_sentence_count >= max_sentences:
                break

    extraction_meta = {
        "pre_cleaning_sentence_count": raw_sentence_count,
        "pre_cleaning_whitespace_token_count": pre_cleaning_whitespace_token_count,
        "pre_cleaning_character_count": pre_cleaning_character_count,
        "skipped_empty_after_tokenization": skipped_empty_after_tokenization,
    }
    return sequences, extraction_meta


def ingest_archive(
    *,
    corpus_id: str,
    language: str,
    family: str,
    role: str,
    source_url: str,
    iso_code: str,
    tokenization_mode: str,
    notes: str = "",
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    raw_dir: Path = DEFAULT_RAW_DIR,
    force_download: bool = False,
    force_output: bool = False,
    max_sentences: int | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    tokens_path = output_dir / f"{corpus_id}_tokens.json"
    manifest_path = output_dir / f"{corpus_id}_manifest.json"
    if tokens_path.exists() and not force_output:
        raise FileExistsError(
            f"{tokens_path} already exists. Use --force-output to overwrite it."
        )

    archive_path = download_archive(source_url, raw_dir=raw_dir, force=force_download)
    sequences, extraction_meta = extract_token_sequences(
        archive_path,
        expected_lang=iso_code,
        tokenization_mode=tokenization_mode,
        max_sentences=max_sentences,
    )
    if not sequences:
        raise ValueError(f"Tatoeba corpus '{corpus_id}' produced zero token sequences")

    stats = corpus_stats(sequences)
    ingest_date = date.today().isoformat()
    archive_hash = sha256_file(archive_path)
    corpus = {
        "corpus_id": corpus_id,
        "language": language,
        "family": family,
        "role": role,
        "source": "tatoeba",
        "source_provider": "Tatoeba",
        "source_url": source_url,
        "download_page": TATOEBA_DOWNLOADS_URL,
        "license_note": TATOEBA_LICENSE_NOTE,
        "ingest_date": ingest_date,
        "archive_path": str(archive_path),
        "archive_sha256": archive_hash,
        "archive_size_bytes": archive_path.stat().st_size,
        "iso_code": iso_code,
        "tokenization": {
            "mode": tokenization_mode,
            "basis": "Tatoeba sentence rows",
            "status": "ready",
        },
        "notes": notes,
        **extraction_meta,
        "sequence_count": stats["sequence_count"],
        "total_tokens": stats["total_tokens"],
        "unique_types": stats["unique_types"],
        "type_token_ratio": stats["type_token_ratio"],
        "mean_seq_length": stats["mean_seq_length"],
        "sequences": sequences,
    }

    with tokens_path.open("w", encoding="utf-8") as fh:
        json.dump(corpus, fh, ensure_ascii=False, indent=2)
    log.info("Wrote Tatoeba tokens to %s", tokens_path)

    manifest = {k: v for k, v in corpus.items() if k != "sequences"}
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
    log.info("Wrote Tatoeba manifest to %s", manifest_path)

    return tokens_path


def ingest_preset(
    corpus_id: str,
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    raw_dir: Path = DEFAULT_RAW_DIR,
    force_download: bool = False,
    force_output: bool = False,
    max_sentences: int | None = None,
) -> Path:
    try:
        preset = TATOEBA_PRESETS[corpus_id]
    except KeyError as exc:
        raise ValueError(
            f"Unknown Tatoeba corpus preset '{corpus_id}'. "
            f"Valid options: {sorted(TATOEBA_PRESETS)}"
        ) from exc

    return ingest_archive(
        corpus_id=preset.corpus_id,
        language=preset.language,
        family=preset.family,
        role=preset.role,
        source_url=preset.source_url,
        iso_code=preset.iso_code,
        tokenization_mode=preset.tokenization_mode,
        notes=preset.notes,
        output_dir=output_dir,
        raw_dir=raw_dir,
        force_download=force_download,
        force_output=force_output,
        max_sentences=max_sentences,
    )


def _cmd_list(args: argparse.Namespace) -> int:
    for preset in TATOEBA_PRESETS.values():
        print(json.dumps(asdict(preset), ensure_ascii=False))
    return 0


def _cmd_ingest(args: argparse.Namespace) -> int:
    tokens_path = ingest_preset(
        args.corpus_id,
        output_dir=args.output_dir,
        raw_dir=args.raw_dir,
        force_download=args.force_download,
        force_output=args.force_output,
        max_sentences=args.max_sentences,
    )
    print(tokens_path)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ingest Tatoeba corpora for RBT v2")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List Tatoeba corpus presets")
    list_parser.set_defaults(func=_cmd_list)

    ingest_parser = subparsers.add_parser("ingest", help="Ingest one Tatoeba preset")
    ingest_parser.add_argument("--corpus-id", required=True, choices=sorted(TATOEBA_PRESETS))
    ingest_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    ingest_parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    ingest_parser.add_argument("--force-download", action="store_true")
    ingest_parser.add_argument("--force-output", action="store_true")
    ingest_parser.add_argument("--max-sentences", type=int, default=None)
    ingest_parser.set_defaults(func=_cmd_ingest)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
