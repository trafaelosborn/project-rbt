"""
Leipzig Corpora Collection ingester for Project RBT v2.

This module downloads a selected Leipzig/Wortschatz archive, extracts the
sentence file, tokenizes each sentence with the existing Project RBT tokenizer,
and writes a v2-scoped corpus plus manifest.

Default outputs:
    data/v2/raw/leipzig/{archive}.tar.gz
    data/v2/corpora/{corpus_id}_tokens.json
    data/v2/corpora/{corpus_id}_manifest.json

Usage:
    python -m src.ingest.leipzig list
    python -m src.ingest.leipzig ingest --corpus-id turkish_news_2024_100k
    python -m src.ingest.leipzig ingest-all
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import tarfile
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
DEFAULT_RAW_DIR = V2_DIR / "raw" / "leipzig"
DEFAULT_OUTPUT_DIR = V2_DIR / "corpora"

LEIPZIG_DOWNLOADS_URL = "https://wortschatz.uni-leipzig.de/en/download/"
LEIPZIG_TERMS_URL = "https://wortschatz.uni-leipzig.de/en/usage"
LEIPZIG_CITATION = (
    "D. Goldhahn, T. Eckart & U. Quasthoff: Building Large Monolingual "
    "Dictionaries at the Leipzig Corpora Collection: From 100 to 200 "
    "Languages. Proceedings of LREC 2012."
)
USER_AGENT = (
    "ProjectRBT/2.0 (v2 corpus ingestion; "
    "https://github.com/spaceranger-press/project-rbt; contact via GitHub)"
)


@dataclass(frozen=True)
class LeipzigCorpusPreset:
    corpus_id: str
    language: str
    iso_code: str
    family: str
    role: str
    source_url: str
    source_type: str
    source_year: str
    sentence_count_label: str
    required_for_h1: bool = True
    default_ingest: bool = True
    tokenization_status: str = "ready"
    notes: str = ""

    @property
    def download_page(self) -> str:
        return f"{LEIPZIG_DOWNLOADS_URL}{self.iso_code}"


LEIPZIG_PRESETS: dict[str, LeipzigCorpusPreset] = {
    "turkish_news_2024_100k": LeipzigCorpusPreset(
        corpus_id="turkish_news_2024_100k",
        language="turkish",
        iso_code="tur",
        family="turkic",
        role="typological_control",
        source_url="https://downloads.wortschatz-leipzig.de/corpora/tur_news_2024_100K.tar.gz",
        source_type="news",
        source_year="2024",
        sentence_count_label="100K",
    ),
    "finnish_news_2022_100k": LeipzigCorpusPreset(
        corpus_id="finnish_news_2022_100k",
        language="finnish",
        iso_code="fin",
        family="uralic",
        role="typological_control",
        source_url="https://downloads.wortschatz-leipzig.de/corpora/fin_news_2022_100K.tar.gz",
        source_type="news",
        source_year="2022",
        sentence_count_label="100K",
    ),
    "vietnamese_news_2022_100k": LeipzigCorpusPreset(
        corpus_id="vietnamese_news_2022_100k",
        language="vietnamese",
        iso_code="vie",
        family="austroasiatic",
        role="typological_control",
        source_url="https://downloads.wortschatz-leipzig.de/corpora/vie_news_2022_100K.tar.gz",
        source_type="news",
        source_year="2022",
        sentence_count_label="100K",
        notes="Registered as the Vietnamese option for the Mandarin-or-Vietnamese control row.",
    ),
    "catalan_news_2022_100k": LeipzigCorpusPreset(
        corpus_id="catalan_news_2022_100k",
        language="catalan",
        iso_code="cat",
        family="romance",
        role="withheld_romance_relative",
        source_url="https://downloads.wortschatz-leipzig.de/corpora/cat_news_2022_100K.tar.gz",
        source_type="news",
        source_year="2022",
        sentence_count_label="100K",
        notes="Chosen over Sardinian for the H1 withheld Romance row because Leipzig has Catalan 100K corpora.",
    ),
    "arabic_news_2022_100k": LeipzigCorpusPreset(
        corpus_id="arabic_news_2022_100k",
        language="arabic",
        iso_code="ara",
        family="semitic",
        role="donor_comparator",
        source_url="https://downloads.wortschatz-leipzig.de/corpora/ara_news_2022_100K.tar.gz",
        source_type="news",
        source_year="2022",
        sentence_count_label="100K",
        notes="Used as the Modern Standard Arabic donor proxy; Leipzig labels the corpus Arabic.",
    ),
    "bulgarian_news_2022_100k": LeipzigCorpusPreset(
        corpus_id="bulgarian_news_2022_100k",
        language="bulgarian",
        iso_code="bul",
        family="slavic",
        role="donor_comparator",
        source_url="https://downloads.wortschatz-leipzig.de/corpora/bul_news_2022_100K.tar.gz",
        source_type="news",
        source_year="2022",
        sentence_count_label="100K",
        notes="Chosen as the default Slavic donor comparator; Serbian remains an alternate.",
    ),
    "german_news_2025_100k": LeipzigCorpusPreset(
        corpus_id="german_news_2025_100k",
        language="german",
        iso_code="deu",
        family="germanic",
        role="donor_comparator",
        source_url="https://downloads.wortschatz-leipzig.de/corpora/deu_news_2025_100K.tar.gz",
        source_type="news",
        source_year="2025",
        sentence_count_label="100K",
        notes="Imperfect Germanic donor proxy for the French contact diagnostic.",
    ),
    "japanese_news_2023_100k": LeipzigCorpusPreset(
        corpus_id="japanese_news_2023_100k",
        language="japanese",
        iso_code="jpn",
        family="japonic",
        role="typological_control",
        source_url="https://downloads.wortschatz-leipzig.de/corpora/jpn_news_2023_100K.tar.gz",
        source_type="news",
        source_year="2023",
        sentence_count_label="100K",
        required_for_h1=False,
        default_ingest=False,
        tokenization_status="needs_tokenization_policy",
        notes=(
            "Deferred post-H1 optional typological control. Not default-ingested "
            "because the current word tokenizer is not a valid Japanese segmentation policy."
        ),
    ),
    "mandarin_wikipedia_2021_100k": LeipzigCorpusPreset(
        corpus_id="mandarin_wikipedia_2021_100k",
        language="mandarin_chinese",
        iso_code="cmn",
        family="sinitic",
        role="typological_control_alternate",
        source_url="https://downloads.wortschatz-leipzig.de/corpora/cmn_wikipedia_2021_100K.tar.gz",
        source_type="wikipedia",
        source_year="2021",
        sentence_count_label="100K",
        required_for_h1=False,
        default_ingest=False,
        tokenization_status="needs_tokenization_policy",
        notes=(
            "Alternate for the Mandarin-or-Vietnamese row. Not default-ingested "
            "until the tokenizer policy for unsegmented Sinitic text is frozen."
        ),
    ),
    "serbian_rs_web_2016_100k": LeipzigCorpusPreset(
        corpus_id="serbian_rs_web_2016_100k",
        language="serbian",
        iso_code="srp",
        family="slavic",
        role="donor_comparator_alternate",
        source_url="https://downloads.wortschatz-leipzig.de/corpora/srp-rs_web_2016_100K.tar.gz",
        source_type="web",
        source_year="2016",
        sentence_count_label="100K",
        required_for_h1=False,
        default_ingest=False,
        notes="Alternate Slavic donor comparator if Bulgarian is rejected during H1 review.",
    ),
    "sardinian_wikipedia_2021_10k": LeipzigCorpusPreset(
        corpus_id="sardinian_wikipedia_2021_10k",
        language="sardinian",
        iso_code="srd",
        family="romance",
        role="withheld_romance_relative_alternate",
        source_url="https://downloads.wortschatz-leipzig.de/corpora/srd_wikipedia_2021_10K.tar.gz",
        source_type="wikipedia",
        source_year="2021",
        sentence_count_label="10K",
        required_for_h1=False,
        default_ingest=False,
        notes="Small alternate for the Catalan-or-Sardinian row; below the main 20K/30K/50K budgets.",
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
    if not name.endswith(".tar.gz"):
        raise ValueError(f"Expected a .tar.gz URL, got: {url}")
    return name


def download_archive(
    url: str,
    raw_dir: Path = DEFAULT_RAW_DIR,
    force: bool = False,
) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    archive_path = raw_dir / archive_name_from_url(url)
    if archive_path.exists() and not force:
        log.info("Using cached Leipzig archive: %s", archive_path)
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
    log.info("Cached Leipzig archive: %s", archive_path)
    return archive_path


def sentence_from_leipzig_line(line: str) -> str:
    line = line.rstrip("\n\r")
    if not line:
        return ""
    parts = line.split("\t", 1)
    if len(parts) == 2 and parts[0].strip().isdigit():
        return parts[1].strip()
    return line.strip()


def _find_sentence_member(tar: tarfile.TarFile) -> tarfile.TarInfo:
    candidates = [
        member
        for member in tar.getmembers()
        if member.isfile() and member.name.endswith("sentences.txt")
    ]
    if not candidates:
        raise ValueError("Archive does not contain a Leipzig sentences.txt file")
    exact = [member for member in candidates if member.name.endswith("-sentences.txt")]
    return sorted(exact or candidates, key=lambda member: member.name)[0]


def extract_token_sequences(
    archive_path: Path,
    max_sentences: int | None = None,
) -> tuple[list[list[str]], dict]:
    sequences: list[list[str]] = []
    raw_sentence_count = 0
    skipped_empty_after_tokenization = 0
    pre_cleaning_whitespace_token_count = 0
    pre_cleaning_character_count = 0

    with tarfile.open(archive_path, "r:gz") as tar:
        sentence_member = _find_sentence_member(tar)
        sentence_file = tar.extractfile(sentence_member)
        if sentence_file is None:
            raise ValueError(f"Could not extract {sentence_member.name}")

        for raw_line in sentence_file:
            sentence = sentence_from_leipzig_line(raw_line.decode("utf-8", errors="replace"))
            if not sentence:
                continue
            raw_sentence_count += 1
            pre_cleaning_whitespace_token_count += len(sentence.split())
            pre_cleaning_character_count += len(sentence)
            tokens = tokenize_sentence(sentence)
            if tokens:
                sequences.append(tokens)
            else:
                skipped_empty_after_tokenization += 1
            if max_sentences is not None and raw_sentence_count >= max_sentences:
                break

    extraction_meta = {
        "sentence_file": sentence_member.name,
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
    iso_code: str | None = None,
    source_type: str | None = None,
    source_year: str | None = None,
    sentence_count_label: str | None = None,
    tokenization_status: str = "ready",
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
        max_sentences=max_sentences,
    )
    if not sequences:
        raise ValueError(f"Leipzig corpus '{corpus_id}' produced zero token sequences")

    stats = corpus_stats(sequences)
    archive_hash = sha256_file(archive_path)
    ingest_date = date.today().isoformat()

    corpus = {
        "corpus_id": corpus_id,
        "language": language,
        "family": family,
        "role": role,
        "source": "leipzig_corpora_collection",
        "source_provider": "Wortschatz Leipzig / Leipzig Corpora Collection",
        "source_url": source_url,
        "download_page": f"{LEIPZIG_DOWNLOADS_URL}{iso_code}" if iso_code else LEIPZIG_DOWNLOADS_URL,
        "terms_url": LEIPZIG_TERMS_URL,
        "license_note": "Downloadable text corpora are listed by Wortschatz Leipzig as CC BY.",
        "citation": LEIPZIG_CITATION,
        "ingest_date": ingest_date,
        "archive_path": str(archive_path),
        "archive_sha256": archive_hash,
        "archive_size_bytes": archive_path.stat().st_size,
        "iso_code": iso_code,
        "source_type": source_type,
        "source_year": source_year,
        "sentence_count_label": sentence_count_label,
        "tokenization": {
            "basis": "Leipzig sentence rows plus Project RBT unicode word tokenizer",
            "mode": "unicode_word",
            "status": tokenization_status,
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
    log.info("Wrote Leipzig tokens to %s", tokens_path)

    manifest = {k: v for k, v in corpus.items() if k != "sequences"}
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
    log.info("Wrote Leipzig manifest to %s", manifest_path)

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
        preset = LEIPZIG_PRESETS[corpus_id]
    except KeyError as exc:
        raise ValueError(
            f"Unknown Leipzig corpus preset '{corpus_id}'. "
            f"Valid options: {sorted(LEIPZIG_PRESETS)}"
        ) from exc

    return ingest_archive(
        corpus_id=preset.corpus_id,
        language=preset.language,
        family=preset.family,
        role=preset.role,
        source_url=preset.source_url,
        iso_code=preset.iso_code,
        source_type=preset.source_type,
        source_year=preset.source_year,
        sentence_count_label=preset.sentence_count_label,
        tokenization_status=preset.tokenization_status,
        notes=preset.notes,
        output_dir=output_dir,
        raw_dir=raw_dir,
        force_download=force_download,
        force_output=force_output,
        max_sentences=max_sentences,
    )


def default_preset_ids(include_tokenization_review: bool = False) -> list[str]:
    return [
        corpus_id
        for corpus_id, preset in LEIPZIG_PRESETS.items()
        if preset.default_ingest
        or (include_tokenization_review and preset.tokenization_status != "ready")
    ]


def _cmd_list(args: argparse.Namespace) -> int:
    for preset in LEIPZIG_PRESETS.values():
        if not args.all and not preset.default_ingest:
            continue
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


def _cmd_ingest_all(args: argparse.Namespace) -> int:
    corpus_ids = args.corpus_id or default_preset_ids(
        include_tokenization_review=args.include_tokenization_review
    )
    written: list[str] = []
    for corpus_id in corpus_ids:
        tokens_path = ingest_preset(
            corpus_id,
            output_dir=args.output_dir,
            raw_dir=args.raw_dir,
            force_download=args.force_download,
            force_output=args.force_output,
            max_sentences=args.max_sentences,
        )
        written.append(str(tokens_path))
    print(json.dumps({"written": written}, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ingest Leipzig corpora for RBT v2")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List Leipzig corpus presets")
    list_parser.add_argument(
        "--all",
        action="store_true",
        help="Include alternates and tokenization-review presets",
    )
    list_parser.set_defaults(func=_cmd_list)

    ingest_parser = subparsers.add_parser("ingest", help="Ingest one Leipzig preset")
    ingest_parser.add_argument("--corpus-id", required=True, choices=sorted(LEIPZIG_PRESETS))
    _add_ingest_args(ingest_parser)
    ingest_parser.set_defaults(func=_cmd_ingest)

    all_parser = subparsers.add_parser("ingest-all", help="Ingest default Leipzig presets")
    all_parser.add_argument(
        "--corpus-id",
        action="append",
        choices=sorted(LEIPZIG_PRESETS),
        help="Restrict ingest-all to one or more preset ids",
    )
    all_parser.add_argument(
        "--include-tokenization-review",
        action="store_true",
        help="Also ingest presets marked as needing a tokenization policy",
    )
    _add_ingest_args(all_parser)
    all_parser.set_defaults(func=_cmd_ingest_all)

    return parser


def _add_ingest_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--force-output", action="store_true")
    parser.add_argument(
        "--max-sentences",
        type=int,
        default=None,
        help="Optional test cap before tokenization",
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
