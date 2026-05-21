"""
Sumerian Corpus Ingester
========================
Purpose:
    Ingest Sumerian cuneiform texts and build a statistical fingerprint for use
    as the structured null model ceiling in the retrodiction pipeline.

    Sumerian is an isolate — no demonstrated genetic relationship with any other
    living or attested language family, including all Indo-European languages.
    It is roughly contemporary with the period in which Latin was developing.
    Its statistical fingerprint therefore represents "real linguistic structure,
    but zero genealogical relationship to Romance."

    Any valid retrodiction of Latin from Romance must score meaningfully closer
    to Latin than to Sumerian.

    Decision log: docs/decisions/009_sumerian_source.md

Data sources (in order of preference):

    PRIMARY — CDLI ATF bulk download:
        URL: https://cdli.mpsa.cnrs.fr/dl/bulk_data/ATF/cdli_atf_20150104.zip
        Format: ATF (ASCII Transliteration Format), plain text
        Content: ~330,000 cuneiform texts, all periods, multiple languages.
        Filter: language marker `#atf: lang sux` (Sumerian) or `#atf: lang sux-x-*`
        The CDLI bulk download is the canonical source cited in the project brief.

        Accessibility note: The CDLI server (cdli.mpsa.cnrs.fr) requires direct
        network access. If unreachable, fall back to the ORACC source below.

    FALLBACK — ORACC DCCLT JSON:
        URL: https://oracc.museum.upenn.edu/json/dcclt.zip
        Format: ORACC CDL JSON (Cuneiform Digital Library JSON schema)
        Content: ~10,000 cuneiform lexical texts; ~3,257 with Sumerian content.
        Genre: Lexical texts (vocabulary lists, not administrative running prose).

        Limitation: Lexical texts have shorter sequences and less grammatical
        structure than administrative texts. The positional and co-occurrence
        statistics are less rich than what CDLI provides. This is documented in
        the manifest and METHODOLOGY.md. The null model is still valid — Sumerian
        lexical structure is genuinely non-IE and statistically real.

ATF format (CDLI):
    Each text block starts with &P{number} = {title}.
    Language is specified by `#atf: lang sux` (or similar).
    Text lines have the format: {line_number}. {tokens...}
    Special tokens:
        {d}godname   — determinative prefix, keep content (strip braces)
        {ki}         — determinative, strip
        [...]        — lacuna (broken), skip
        <...>        — editorial correction, keep content
        ...          — lacuna, skip
        /            — line continuation, skip
    Numerals: keep as tokens (they reflect Sumerian numerical system structure)

ORACC CDL JSON format:
    Recursive tree with "cdl" arrays. Terminal nodes:
        node="l": lemma, with "f": {"form": "surface_form", "lang": "sux"}
    Extract: f.form where f.lang starts with "sux" and form is not empty.

Outputs:
    data/processed/nulls/sumerian/sumerian_tokens.json    — tokenized corpus
    data/processed/nulls/sumerian/sumerian_manifest.json  — retrieval metadata

Usage:
    python -m src.ingest.sumerian                   # try CDLI, fall back to ORACC
    python -m src.ingest.sumerian --source cdli     # CDLI only (fail if unavailable)
    python -m src.ingest.sumerian --source oracc    # ORACC DCCLT only
    python -m src.ingest.sumerian --source oracc --max-texts 1000
"""

import argparse
import io
import json
import logging
import re
import time
import zipfile
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "nulls" / "sumerian"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CDLI_ATF_URL = "https://cdli.mpsa.cnrs.fr/dl/bulk_data/ATF/cdli_atf_20150104.zip"
ORACC_DCCLT_URL = "https://oracc.museum.upenn.edu/json/dcclt.zip"

RATE_LIMIT_DELAY = 0.0   # both sources are single bulk downloads
REQUEST_TIMEOUT = 300    # 5 minutes for large downloads

USER_AGENT = (
    "ProjectRBT/1.0 (Sumerian null model ingestion; "
    "https://github.com/spaceranger-press/project-rbt)"
)

# ATF language marker pattern for Sumerian
_SUMERIAN_LANG_RE = re.compile(r"#atf:\s+lang\s+sux", re.IGNORECASE)

# ATF line number: digits followed by period (may include letters: 1', o.1, etc.)
_LINE_NUM_RE = re.compile(r"^\d[\d']*\.\s+")

# ATF tokens to strip or skip
_LACUNA_RE = re.compile(r"\[.*?\]|\.\.\.")          # [...] or ...
_DETERMINATIVE_RE = re.compile(r"\{[^}]+\}")        # {d}, {ki}, etc.
_EDITORIAL_RE = re.compile(r"<([^>]+)>")            # <correction> → keep content
_COMMENT_RE = re.compile(r"#.*$")                   # inline comments


# ---------------------------------------------------------------------------
# CDLI ATF parser
# ---------------------------------------------------------------------------

def _parse_atf_line(raw: str) -> list[str]:
    """
    Parse one ATF text line into a list of token strings.

    Steps:
        1. Strip the line number prefix.
        2. Remove inline comments.
        3. Replace lacunae ([...] and ...) with nothing.
        4. Strip determinative braces: {d}enlil → enlil.
        5. Keep editorial corrections: <correction> → correction.
        6. Split on whitespace and filter empty tokens.
    """
    # Strip line number
    line = _LINE_NUM_RE.sub("", raw).strip()
    # Remove inline comments
    line = _COMMENT_RE.sub("", line).strip()
    # Remove lacunae
    line = _LACUNA_RE.sub("", line)
    # Strip determinatives (keep content)
    line = _DETERMINATIVE_RE.sub(
        lambda m: m.group(0)[1:-1].split("}")[0],  # strip braces, keep inner
        line,
    )
    # Keep editorial corrections
    line = _EDITORIAL_RE.sub(r"\1", line)
    # Split and filter
    tokens = [t.strip("-").strip() for t in line.split() if t.strip("-").strip()]
    return [t for t in tokens if t]


def ingest_cdli(
    session: requests.Session,
    max_texts: int | None = None,
) -> tuple[list[list[str]], dict]:
    """
    Download and parse the CDLI ATF bulk file.

    Args:
        session:    requests.Session to use.
        max_texts:  Maximum Sumerian texts to process (None = all).

    Returns:
        (sequences, stats_dict)
    """
    log.info("Downloading CDLI ATF bulk file from %s ...", CDLI_ATF_URL)
    resp = session.get(CDLI_ATF_URL, timeout=REQUEST_TIMEOUT, stream=True)
    resp.raise_for_status()

    # Stream into memory buffer
    buf = io.BytesIO()
    total = 0
    for chunk in resp.iter_content(chunk_size=1024 * 1024):
        buf.write(chunk)
        total += len(chunk)
        if total % (10 * 1024 * 1024) == 0:
            log.info("Downloaded %d MB...", total // (1024 * 1024))

    log.info("Download complete: %d MB", total // (1024 * 1024))
    buf.seek(0)

    sequences: list[list[str]] = []
    texts_processed = 0
    texts_seen = 0
    current_is_sumerian = False
    current_sequence: list[str] = []

    with zipfile.ZipFile(buf) as zf:
        # The zip likely contains a single ATF file
        atf_files = [n for n in zf.namelist() if n.endswith(".atf")]
        if not atf_files:
            raise ValueError("No .atf files found in CDLI zip")
        log.info("ATF files in zip: %s", atf_files)

        for atf_name in atf_files:
            with zf.open(atf_name) as fh:
                for raw_line in fh:
                    line = raw_line.decode("utf-8", errors="replace").rstrip()

                    if line.startswith("&P"):
                        # New text block: flush previous
                        if current_is_sumerian and current_sequence:
                            sequences.append(current_sequence)
                        if current_is_sumerian:
                            texts_processed += 1
                        current_is_sumerian = False
                        current_sequence = []
                        texts_seen += 1

                        if max_texts is not None and texts_processed >= max_texts:
                            break

                    elif _SUMERIAN_LANG_RE.match(line):
                        current_is_sumerian = True

                    elif current_is_sumerian and _LINE_NUM_RE.match(line):
                        tokens = _parse_atf_line(line)
                        current_sequence.extend(tokens)

                else:
                    # End of file: flush last text
                    if current_is_sumerian and current_sequence:
                        sequences.append(current_sequence)
                        texts_processed += 1
                    continue
                break

    log.info(
        "CDLI: %d texts seen, %d Sumerian texts processed, %d sequences",
        texts_seen, texts_processed, len(sequences),
    )
    return sequences, {
        "source": "cdli_atf",
        "cdli_url": CDLI_ATF_URL,
        "texts_processed": texts_processed,
    }


# ---------------------------------------------------------------------------
# ORACC DCCLT JSON parser
# ---------------------------------------------------------------------------

def _collect_oracc_forms(node: dict, out: list[str]) -> None:
    """Recursively collect Sumerian surface forms from an ORACC CDL node."""
    if not isinstance(node, dict):
        return
    if node.get("node") == "l":
        f = node.get("f", {})
        lang = f.get("lang", "")
        form = f.get("form", "")
        if lang.startswith("sux") and form and form != "x":
            out.append(form)
    for child in node.get("cdl", []):
        _collect_oracc_forms(child, out)


def ingest_oracc(
    session: requests.Session,
    max_texts: int | None = None,
    local_zip_path: Path | None = None,
) -> tuple[list[list[str]], dict]:
    """
    Download and parse the ORACC DCCLT JSON corpus.

    Args:
        session:        requests.Session to use.
        max_texts:      Maximum texts to process.
        local_zip_path: If provided, read from this local ZIP instead of downloading.

    Returns:
        (sequences, stats_dict)
    """
    if local_zip_path is not None:
        log.info("Using local ORACC DCCLT ZIP: %s", local_zip_path)
        buf = io.BytesIO(local_zip_path.read_bytes())
    else:
        log.info("Downloading ORACC DCCLT corpus from %s ...", ORACC_DCCLT_URL)
        resp = session.get(ORACC_DCCLT_URL, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        buf = io.BytesIO(resp.content)
        log.info("Download complete: %.1f MB", len(resp.content) / (1024 * 1024))

    sequences: list[list[str]] = []
    texts_processed = 0
    texts_skipped = 0

    with zipfile.ZipFile(buf) as zf:
        # Load catalogue to identify Sumerian texts
        with zf.open("dcclt/catalogue.json") as f:
            catalogue = json.loads(f.read().decode("utf-8", "replace"))

        sux_ids = [
            pid for pid, meta in catalogue.get("members", {}).items()
            if "sumerian" in meta.get("language", "").lower()
        ]
        log.info("Sumerian texts in catalogue: %d", len(sux_ids))

        if max_texts is not None:
            sux_ids = sux_ids[:max_texts]

        all_names = set(zf.namelist())

        for pid in sux_ids:
            corpus_path = f"dcclt/corpusjson/{pid}.json"
            if corpus_path not in all_names:
                texts_skipped += 1
                continue
            try:
                with zf.open(corpus_path) as f:
                    obj = json.loads(f.read().decode("utf-8", "replace"))
                forms: list[str] = []
                _collect_oracc_forms({"cdl": obj.get("cdl", [])}, forms)
                if forms:
                    sequences.append(forms)
                    texts_processed += 1
                else:
                    texts_skipped += 1
            except Exception as exc:
                log.debug("Skipping %s: %s", pid, exc)
                texts_skipped += 1

    log.info(
        "ORACC DCCLT: %d texts processed, %d skipped, %d sequences",
        texts_processed, texts_skipped, len(sequences),
    )
    return sequences, {
        "source": "oracc_dcclt",
        "oracc_url": ORACC_DCCLT_URL,
        "oracc_project": "dcclt",
        "genre": "lexical",
        "texts_processed": texts_processed,
        "texts_skipped": texts_skipped,
        "limitation": (
            "DCCLT contains lexical texts (vocabulary lists), not administrative "
            "or literary running prose. Sequences are shorter and less grammatically "
            "structured than CDLI administrative texts. This is a known limitation "
            "of the fallback source. See docs/decisions/009_sumerian_source.md."
        ),
    }


# ---------------------------------------------------------------------------
# Write output
# ---------------------------------------------------------------------------

def _write_output(
    sequences: list[list[str]],
    source_meta: dict,
    output_dir: Path,
) -> Path:
    from datetime import date
    output_dir.mkdir(parents=True, exist_ok=True)

    total = sum(len(s) for s in sequences)
    types = set(t for seq in sequences for t in seq)
    ttr = len(types) / total if total > 0 else 0.0

    corpus = {
        "language": "sumerian",
        **source_meta,
        "fetch_date": date.today().isoformat(),
        "sequence_count": len(sequences),
        "total_tokens": total,
        "unique_types": len(types),
        "type_token_ratio": ttr,
        "mean_seq_length": total / len(sequences) if sequences else 0.0,
        "null_model_role": "structured_non_ie_ceiling",
        "sequences": sequences,
    }

    tokens_path = output_dir / "sumerian_tokens.json"
    manifest_path = output_dir / "sumerian_manifest.json"

    with tokens_path.open("w", encoding="utf-8") as fh:
        json.dump(corpus, fh, ensure_ascii=False, indent=2)
    log.info("Wrote Sumerian corpus to %s", tokens_path)

    manifest = {k: v for k, v in corpus.items() if k != "sequences"}
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)

    log.info(
        "Sumerian corpus: %d sequences, %d tokens, %d types, TTR=%.4f",
        len(sequences), total, len(types), ttr,
    )
    return tokens_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(
    source: str = "auto",
    max_texts: int | None = None,
    output_dir: Path = OUTPUT_DIR,
    local_zip_path: Path | None = None,
) -> Path:
    """
    Ingest Sumerian corpus from the specified source.

    Args:
        source:         "auto" (try CDLI, fall back to ORACC),
                        "cdli" (CDLI only),
                        "oracc" (ORACC DCCLT only).
        max_texts:      Maximum number of Sumerian texts to process.
        output_dir:     Output directory.
        local_zip_path: Path to a locally cached ORACC DCCLT ZIP (skips download).

    Returns:
        Path to the written tokens JSON file.
    """
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    if source in ("auto", "cdli"):
        try:
            sequences, meta = ingest_cdli(session, max_texts)
            return _write_output(sequences, meta, output_dir)
        except Exception as exc:
            if source == "cdli":
                raise
            log.warning(
                "CDLI ingestion failed (%s). Falling back to ORACC DCCLT.", exc
            )

    sequences, meta = ingest_oracc(session, max_texts, local_zip_path=local_zip_path)
    return _write_output(sequences, meta, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest Sumerian null model corpus")
    parser.add_argument(
        "--source",
        choices=["auto", "cdli", "oracc"],
        default="auto",
        help="Data source: auto (try CDLI, fall back to ORACC), cdli, or oracc",
    )
    parser.add_argument(
        "--max-texts",
        type=int,
        default=None,
        help="Maximum Sumerian texts to process (default: all)",
    )
    parser.add_argument(
        "--local-zip",
        type=Path,
        default=None,
        help="Path to a locally cached ORACC DCCLT ZIP file (skips download)",
    )
    args = parser.parse_args()
    run(source=args.source, max_texts=args.max_texts, local_zip_path=args.local_zip)
