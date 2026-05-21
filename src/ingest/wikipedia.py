"""
Wikipedia API Ingester
======================
Purpose:
    Fetch article text from a Wikipedia language edition via the MediaWiki API,
    tokenize on the fly, and write the resulting token sequences to a processed
    corpus file. Raw article text is never written to disk.

Outputs:
    data/processed/romance/{language}_tokens.json   — tokenized corpus
    data/processed/romance/{language}_manifest.json — retrieval metadata

Output format (tokens file):
    {
      "language": "french",
      "wiki_code": "fr",
      "source": "wikipedia_api",
      "api_date": "2026-04-07",
      "article_count": 500,
      "total_tokens": 234567,
      "sequence_count": 18432,
      "sequences": [
        ["le", "président", "de", "la", "république"],
        ["il", "a", "déclaré", "que"],
        ...
      ]
    }

API strategy:
    Uses action=query with generator=random (rnnamespace=0) to draw random
    articles, then action=query&prop=extracts to retrieve plain-text extracts.
    Articles are fetched in batches of BATCH_SIZE. Processing is streaming:
    each batch is tokenized and appended to the accumulator before the next
    request is made. Memory usage is O(processed corpus), not O(raw downloads).

    Decision log: docs/decisions/005_wikipedia_api_strategy.md

Rate limiting:
    Wikipedia's API rate limit for anonymous clients is approximately 200 requests
    per minute. This ingester sleeps RATE_LIMIT_DELAY seconds between batches and
    sets a User-Agent header identifying the project, as required by the Wikimedia
    API terms of service.

Corpus size target:
    DEFAULT_ARTICLE_COUNT articles per language. This is a deliberate choice:
    large enough for stable statistical fingerprints, small enough to run without
    a multi-hour download. Thin-corpus languages (Occitan, Genoese) may yield
    fewer articles than requested if their Wikipedia is smaller.

    Decision log: docs/decisions/006_corpus_size_target.md

Assumptions:
    - Wikipedia API returns UTF-8 JSON.
    - The 'extract' field from prop=extracts contains clean prose (no wiki markup
      after exintro=false&explaintext=true). Some residual artifacts (section
      headers that survived stripping) are handled by the tokenizer.
    - Articles with very short extracts (< MIN_EXTRACT_CHARS) are skipped.

Known limitations:
    - Random sampling does not guarantee representative topic coverage.
      Wikipedia's article distribution is skewed toward certain domains.
      This is acceptable for RBT: register consistency (formal encyclopedic prose)
      is more important than topic diversity. Documented in METHODOLOGY.md.
    - Occitan (oc) and Genoese (lij) Wikipedias are thin. Article count and total
      token count are recorded in the manifest; downstream weighting must account
      for this.

Usage:
    python -m src.ingest.wikipedia --language french
    python -m src.ingest.wikipedia --language italian --articles 1000
    python -m src.ingest.wikipedia --language portuguese --sequester
"""

import argparse
import json
import logging
import time
from datetime import date
from pathlib import Path

import requests

from src.ingest.tokenize import tokenize_text, corpus_stats

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
ROMANCE_DIR = PROCESSED_DIR / "romance"
SEQUESTERED_DIR = PROJECT_ROOT / "data" / "sequestered"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Wikipedia language codes for each corpus.
LANGUAGE_WIKI_CODES: dict[str, str] = {
    "italian":    "it",
    "french":     "fr",
    "spanish":    "es",
    "romanian":   "ro",
    "occitan":    "oc",
    "genoese":    "lij",
    "portuguese": "pt",   # positive control — sequestered on ingest
}

DEFAULT_ARTICLE_COUNT = 500
BATCH_SIZE = 20           # articles per API request
MIN_EXTRACT_CHARS = 200   # skip articles with fewer characters than this
RATE_LIMIT_DELAY = 0.5    # seconds between batch requests

USER_AGENT = (
    "ProjectRBT/1.0 (Romance retrodiction research; "
    "https://github.com/spaceranger-press/project-rbt; contact via GitHub)"
)

# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def _api_url(wiki_code: str) -> str:
    return f"https://{wiki_code}.wikipedia.org/w/api.php"


def _fetch_random_extracts(
    wiki_code: str,
    session: requests.Session,
    count: int,
) -> list[str]:
    """
    Fetch `count` random articles with plain-text extracts in a single API call.
    Uses generator=random with prop=extracts — one round-trip instead of two.
    Returns a list of extract strings (only non-empty ones).
    """
    url = _api_url(wiki_code)
    params = {
        "action": "query",
        "generator": "random",
        "grnnamespace": 0,
        "grnlimit": min(count, 20),   # extracts API reliable up to 20 per request
        "prop": "extracts",
        "explaintext": True,
        "exsectionformat": "plain",
        "format": "json",
    }
    resp = session.get(url, params=params, timeout=60)
    resp.raise_for_status()
    pages = resp.json().get("query", {}).get("pages", {})
    return [page.get("extract", "") or "" for page in pages.values()]


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

def ingest(
    language: str,
    article_count: int = DEFAULT_ARTICLE_COUNT,
    output_dir: Path | None = None,
    sequester: bool = False,
) -> Path:
    """
    Fetch articles from Wikipedia, tokenize, and write corpus file.

    Args:
        language:      Language name (key in LANGUAGE_WIKI_CODES).
        article_count: Target number of articles to process.
        output_dir:    Override default output directory.
        sequester:     If True, write output to data/sequestered/{language}/
                       instead of data/processed/romance/. Used for Portuguese.

    Returns:
        Path to the written tokens JSON file.
    """
    if language not in LANGUAGE_WIKI_CODES:
        raise ValueError(
            f"Unknown language '{language}'. "
            f"Valid options: {sorted(LANGUAGE_WIKI_CODES)}"
        )

    wiki_code = LANGUAGE_WIKI_CODES[language]

    if output_dir is None:
        if sequester:
            output_dir = SEQUESTERED_DIR / language
        else:
            output_dir = ROMANCE_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    if sequester:
        log.warning(
            "SEQUESTRATION: %s corpus will be written to %s. "
            "This corpus must never be used as reconstruction input.",
            language.upper(),
            output_dir,
        )

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    all_sequences: list[list[str]] = []
    articles_processed = 0
    articles_skipped = 0
    batches_fetched = 0

    log.info(
        "Starting ingestion: language=%s, wiki=%s, target=%d articles",
        language, wiki_code, article_count,
    )

    while articles_processed < article_count:
        remaining = article_count - articles_processed
        batch_target = min(BATCH_SIZE, remaining)

        try:
            extracts = _fetch_random_extracts(wiki_code, session, batch_target)
            if not extracts:
                log.warning("No extracts returned — Wikipedia may be rate-limiting. Stopping.")
                break
        except requests.RequestException as exc:
            log.error("API error on batch %d: %s — stopping ingestion", batches_fetched + 1, exc)
            break

        for extract in extracts:
            if len(extract) < MIN_EXTRACT_CHARS:
                articles_skipped += 1
                continue
            sequences = tokenize_text(extract)
            if sequences:
                all_sequences.extend(sequences)
                articles_processed += 1

        batches_fetched += 1
        log.info(
            "Batch %d: processed=%d, skipped=%d, sequences=%d",
            batches_fetched, articles_processed, articles_skipped, len(all_sequences),
        )

        if articles_processed < article_count:
            time.sleep(RATE_LIMIT_DELAY)

    stats = corpus_stats(all_sequences)
    log.info(
        "Ingestion complete: %d articles, %d sequences, %d tokens, TTR=%.4f",
        articles_processed,
        stats["sequence_count"],
        stats["total_tokens"],
        stats["type_token_ratio"],
    )

    api_date = date.today().isoformat()

    tokens_path = output_dir / f"{language}_tokens.json"
    manifest_path = output_dir / f"{language}_manifest.json"

    corpus = {
        "language": language,
        "wiki_code": wiki_code,
        "source": "wikipedia_api",
        "api_date": api_date,
        "article_count": articles_processed,
        "articles_skipped": articles_skipped,
        "total_tokens": stats["total_tokens"],
        "sequence_count": stats["sequence_count"],
        "unique_types": stats["unique_types"],
        "type_token_ratio": stats["type_token_ratio"],
        "mean_seq_length": stats["mean_seq_length"],
        "sequestered": sequester,
        "sequences": all_sequences,
    }

    with tokens_path.open("w", encoding="utf-8") as fh:
        json.dump(corpus, fh, ensure_ascii=False, indent=2)
    log.info("Wrote tokens to %s", tokens_path)

    manifest = {k: v for k, v in corpus.items() if k != "sequences"}
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
    log.info("Wrote manifest to %s", manifest_path)

    return tokens_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest Wikipedia corpus for a Romance language")
    parser.add_argument(
        "--language",
        required=True,
        choices=sorted(LANGUAGE_WIKI_CODES),
        help="Language to ingest",
    )
    parser.add_argument(
        "--articles",
        type=int,
        default=DEFAULT_ARTICLE_COUNT,
        help=f"Number of articles to fetch (default: {DEFAULT_ARTICLE_COUNT})",
    )
    parser.add_argument(
        "--sequester",
        action="store_true",
        help="Write output to sequestered directory (use for Portuguese)",
    )
    args = parser.parse_args()
    ingest(language=args.language, article_count=args.articles, sequester=args.sequester)
