"""
Latin Corpus Ingester (Perseus Digital Library)
================================================
Purpose:
    Fetch classical Latin texts from the Perseus Digital Library canonical corpus
    on GitHub (PerseusDL/canonical-latinLit), tokenize, and write to the
    sequestered Island B directory. Raw XML is never written to disk.

    THIS MODULE WRITES TO THE SEQUESTERED DIRECTORY.
    The sequestration guard is NOT enforced here — this module's job is to
    populate the sequestered store. The guard blocks READ access from
    reconstruction modules, not WRITE access during ingestion.

Corpus source:
    Repository: https://github.com/PerseusDL/canonical-latinLit
    Format: TEI P5 XML
    License: CC BY-SA 4.0

    The repository contains 428 Latin XML files covering major classical authors:
    Caesar, Cicero, Virgil, Livy, Sallust, Tacitus, Ovid, Pliny, and others.
    Files are identified by a naming convention: *-lat*.xml.

    Decision log: docs/decisions/008_latin_corpus_source.md

API strategy:
    1. Fetch the full file tree from the GitHub API (1 request).
    2. Filter for Latin text files by filename pattern.
    3. Fetch each file from raw.githubusercontent.com (no API rate limit).
    4. Parse TEI XML to extract prose elements (<p>, <l>, <ab>).
    5. Tokenize using the same Unicode tokenizer as Romance corpora.
    6. Write to data/sequestered/latin/.

    GitHub API rate limit (unauthenticated): 60 requests/hour for /api/ endpoints.
    raw.githubusercontent.com is not subject to this limit.
    A small delay between raw file fetches is applied as a courtesy.

Text extraction:
    - TEI namespace: http://www.tei-c.org/ns/1.0
    - Text-bearing elements: <p>, <l> (verse line), <ab> (anonymous block)
    - Content extracted with itertext() — strips all TEI markup tags, keeping
      only character data.
    - Elements inside <teiHeader> are excluded (metadata, not text).

Outputs:
    data/sequestered/latin/latin_tokens.json    — tokenized corpus
    data/sequestered/latin/latin_manifest.json  — retrieval metadata

Output format:
    Same schema as Wikipedia corpus files:
    {
      "language": "latin",
      "source": "perseus_canonical",
      "github_repo": "PerseusDL/canonical-latinLit",
      "fetch_date": "2026-04-07",
      "file_count": 428,
      "files_processed": 412,
      "sequence_count": 87345,
      "total_tokens": 2341567,
      "sequences": [[tok, tok, ...], ...],
      "sequestered": true
    }

Assumptions:
    - All files matching the Latin filename pattern contain Latin text.
      A small number may be bilingual or contain Greek quotations; these
      are included (the tokenizer handles mixed scripts gracefully).
    - Register is formal classical Latin throughout — Cicero, Caesar, Virgil,
      Livy, Sallust, etc. This matches the corpus description in METHODOLOGY.md.

Known limitations:
    - Verse texts (Virgil, Ovid, Horace) have shorter sequences than prose.
      Mixed-genre corpus is intentional: provides variance in sequence structure.
    - Some files contain Late Latin or Medieval Latin texts from the corpus.
      Not filtered; included as natural extension of the Latin register.
    - 428 files over raw.githubusercontent.com with RATE_LIMIT_DELAY seconds
      between requests takes approximately 10–15 minutes.

Usage:
    python -m src.ingest.latin
    python -m src.ingest.latin --max-files 50   # quick test with subset
"""

import argparse
import json
import logging
import time
import xml.etree.ElementTree as ET
from datetime import date
from pathlib import Path

import requests

from src.ingest.tokenize import tokenize_text, corpus_stats
from src.sequester.guard import sequestered_path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GITHUB_API_TREE = (
    "https://api.github.com/repos/PerseusDL/canonical-latinLit/"
    "git/trees/master?recursive=1"
)
RAW_BASE = "https://raw.githubusercontent.com/PerseusDL/canonical-latinLit/master/"

TEI_NS = "http://www.tei-c.org/ns/1.0"
TEXT_TAGS = {f"{{{TEI_NS}}}{tag}" for tag in ("p", "l", "ab")}
HEADER_TAG = f"{{{TEI_NS}}}teiHeader"

RATE_LIMIT_DELAY = 0.3   # seconds between raw file fetches
REQUEST_TIMEOUT = 60
MAX_FILES_DEFAULT = None  # None = all files

USER_AGENT = (
    "ProjectRBT/1.0 (Latin corpus ingestion; "
    "https://github.com/spaceranger-press/project-rbt)"
)

# ---------------------------------------------------------------------------
# File listing
# ---------------------------------------------------------------------------

def fetch_latin_file_paths(session: requests.Session) -> list[str]:
    """
    Fetch the repo tree from GitHub API and return paths of Latin XML files.
    Returns paths relative to repo root (e.g. 'data/phi0448/phi001/...xml').
    """
    log.info("Fetching canonical-latinLit file tree from GitHub API...")
    resp = session.get(GITHUB_API_TREE, timeout=30)
    resp.raise_for_status()
    tree = resp.json()

    paths = [
        entry["path"]
        for entry in tree.get("tree", [])
        if entry["type"] == "blob"
        and entry["path"].endswith(".xml")
        and "-lat" in entry["path"]
    ]
    log.info("Found %d Latin XML files", len(paths))
    return paths


# ---------------------------------------------------------------------------
# TEI XML extraction
# ---------------------------------------------------------------------------

def extract_text_from_tei(xml_bytes: bytes) -> list[str]:
    """
    Parse a TEI XML file and extract text from prose/verse elements.
    Returns a list of text strings (one per <p>/<l>/<ab> element in the body).
    Skips elements inside <teiHeader>.
    """
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as exc:
        log.warning("XML parse error: %s", exc)
        return []

    texts = []
    header_seen = False

    for elem in root.iter():
        if elem.tag == HEADER_TAG:
            header_seen = True
        if header_seen and elem.tag == HEADER_TAG:
            # Once we've entered the header, skip until body
            continue
        if elem.tag in TEXT_TAGS:
            text = "".join(elem.itertext()).strip()
            if text:
                texts.append(text)

    return texts


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

def ingest(
    max_files: int | None = MAX_FILES_DEFAULT,
    output_dir: Path | None = None,
) -> Path:
    """
    Fetch Latin texts from Perseus, tokenize, and write to Island B.

    Args:
        max_files:  Maximum number of XML files to process. None = all.
                    Use a small number (e.g. 10) for testing.
        output_dir: Override default sequestered output directory.

    Returns:
        Path to the written tokens JSON file.
    """
    if output_dir is None:
        output_dir = sequestered_path("latin")
    output_dir.mkdir(parents=True, exist_ok=True)

    log.warning(
        "ISLAND B INGESTION: Writing Latin corpus to sequestered directory %s. "
        "This corpus must never be used as reconstruction input.",
        output_dir,
    )

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    file_paths = fetch_latin_file_paths(session)
    if max_files is not None:
        file_paths = file_paths[:max_files]
        log.info("Limited to first %d files (--max-files)", max_files)

    all_sequences: list[list[str]] = []
    files_processed = 0
    files_errored = 0

    for i, path in enumerate(file_paths):
        url = RAW_BASE + path
        try:
            resp = session.get(url, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            xml_bytes = resp.content
        except requests.RequestException as exc:
            log.warning("Failed to fetch %s: %s", path, exc)
            files_errored += 1
            if i < len(file_paths) - 1:
                time.sleep(RATE_LIMIT_DELAY)
            continue

        texts = extract_text_from_tei(xml_bytes)
        sequences = []
        for text in texts:
            sequences.extend(tokenize_text(text))

        all_sequences.extend(sequences)
        files_processed += 1

        if (i + 1) % 20 == 0:
            log.info(
                "Progress: %d/%d files, %d sequences so far",
                i + 1, len(file_paths), len(all_sequences),
            )

        if i < len(file_paths) - 1:
            time.sleep(RATE_LIMIT_DELAY)

    stats = corpus_stats(all_sequences)
    log.info(
        "Ingestion complete: %d/%d files, %d sequences, %d tokens, TTR=%.4f",
        files_processed,
        len(file_paths),
        stats["sequence_count"],
        stats["total_tokens"],
        stats["type_token_ratio"],
    )

    fetch_date = date.today().isoformat()
    tokens_path = output_dir / "latin_tokens.json"
    manifest_path = output_dir / "latin_manifest.json"

    corpus = {
        "language": "latin",
        "source": "perseus_canonical",
        "github_repo": "PerseusDL/canonical-latinLit",
        "fetch_date": fetch_date,
        "file_count": len(file_paths),
        "files_processed": files_processed,
        "files_errored": files_errored,
        "sequence_count": stats["sequence_count"],
        "total_tokens": stats["total_tokens"],
        "unique_types": stats["unique_types"],
        "type_token_ratio": stats["type_token_ratio"],
        "mean_seq_length": stats["mean_seq_length"],
        "sequestered": True,
        "sequences": all_sequences,
    }

    with tokens_path.open("w", encoding="utf-8") as fh:
        json.dump(corpus, fh, ensure_ascii=False, indent=2)
    log.info("Wrote Island B Latin corpus to %s", tokens_path)

    manifest = {k: v for k, v in corpus.items() if k != "sequences"}
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
    log.info("Wrote manifest to %s", manifest_path)

    return tokens_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingest Latin corpus from Perseus Digital Library → Island B"
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Limit number of XML files processed (default: all ~428)",
    )
    args = parser.parse_args()
    ingest(max_files=args.max_files)
