"""
N-gram Profile Builder
======================
Purpose:
    Build bigram and trigram frequency profiles from a tokenized language corpus.
    This is one of four statistical fingerprint components used in the
    retrodiction pipeline.

    N-gram profiles capture sequential token patterns at short range. Bigrams encode
    adjacent-word collocations (a signal of phrase structure and morphological pattern);
    trigrams encode local trigram constructions. The profile is stored as a normalized
    frequency distribution over the top-N most common n-grams, enabling comparison
    across corpora of different sizes.

Inputs:
    data/processed/romance/{language}_tokens.json   — tokenized corpus records

Outputs:
    data/matrices/{language}_ngram_meta.json  — top-N bigrams and trigrams with normalized
                                                frequencies; no .npy file (profiles are
                                                sparse and JSON-serializable at this size)

Profile definition:
    For each n in {2, 3}:
        - Count all n-gram occurrences across all sequences.
        - Keep the top TOP_N most frequent n-grams.
        - Normalize counts to relative frequencies (sum to 1.0 over kept n-grams).
        - Store as {n-gram string: relative frequency} dict.

    The n-gram string format is tokens joined by " | " to avoid ambiguity with
    tokens that contain spaces.

Parameters:
    TOP_N (int, default=5000): number of most frequent n-grams to retain per order.

    Decision log: docs/decisions/003_ngram_top_n.md

Assumptions:
    - Sequence boundaries are hard: no n-gram spans two sequences.
    - Bigrams and trigrams only. Higher orders are out of scope for Phase 2.

Known limitations:
    - TOP_N=5000 may capture nearly all bigrams for thin corpora (Occitan, Genoese).
      The actual count is recorded in metadata.
    - Type/token ratio is included here rather than as a separate module since it
      requires the same vocabulary counting pass.

Validation:
    - Bigram relative frequencies must sum to approximately 1.0.
    - Trigram relative frequencies must sum to approximately 1.0.
    - TOP_N must not exceed actual unique n-gram count (capped silently).

Usage:
    python -m src.fingerprint.ngram --language italian
    python -m src.fingerprint.ngram --language french --top-n 3000
"""

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MATRICES_DIR = PROJECT_ROOT / "data" / "matrices"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_TOP_N = 5000
NGRAM_SEP = " | "  # separator for n-gram string keys


# ---------------------------------------------------------------------------
# N-gram counting
# ---------------------------------------------------------------------------

def extract_ngrams(sequences: list[list[str]], n: int) -> Counter:
    """
    Extract all n-grams from a list of token sequences.
    No n-gram spans a sequence boundary.

    Returns a Counter of (tok1, tok2, ...) tuples.
    """
    counter: Counter = Counter()
    for seq in sequences:
        for i in range(len(seq) - n + 1):
            gram = tuple(seq[i : i + n])
            counter[gram] += 1
    return counter


def build_profile(counter: Counter, top_n: int) -> dict[str, float]:
    """
    Build a normalized frequency profile from a raw n-gram Counter.

    Returns a dict mapping n-gram string → relative frequency,
    covering the top_n most frequent n-grams.
    """
    top = counter.most_common(top_n)
    total = sum(count for _, count in top)
    if total == 0:
        return {}
    return {NGRAM_SEP.join(gram): count / total for gram, count in top}


# ---------------------------------------------------------------------------
# Type/token ratio
# ---------------------------------------------------------------------------

def type_token_ratio(sequences: list[list[str]]) -> float:
    """
    Compute the type/token ratio: unique token types / total token count.

    A higher TTR indicates greater lexical diversity (less repetition).
    Synthetic languages tend toward lower TTR than analytic languages for
    equal-length texts, because inflectional morphology generates more
    distinct surface forms.

    Returns 0.0 for empty corpora.
    """
    total = 0
    types: set[str] = set()
    for seq in sequences:
        total += len(seq)
        types.update(seq)
    if total == 0:
        return 0.0
    return len(types) / total


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_profile(profile: dict[str, float], name: str) -> None:
    if not profile:
        log.warning("%s profile is empty", name)
        return
    total = sum(profile.values())
    if not (0.999 <= total <= 1.001):
        raise ValueError(f"{name} profile frequencies sum to {total:.6f}, expected ~1.0")
    log.info("%s profile validated: %d entries, sum=%.6f", name, len(profile), total)


# ---------------------------------------------------------------------------
# Write outputs
# ---------------------------------------------------------------------------

def write_metadata(
    bigram_profile: dict[str, float],
    trigram_profile: dict[str, float],
    ttr: float,
    params: dict,
    path: Path,
) -> None:
    meta = {
        "pipeline_phase": "P2",
        "language": params.get("language", "unknown"),
        "top_n": params.get("top_n", DEFAULT_TOP_N),
        "ngram_separator": NGRAM_SEP,
        "type_token_ratio": ttr,
        "bigram_count": len(bigram_profile),
        "trigram_count": len(trigram_profile),
        "bigrams": bigram_profile,
        "trigrams": trigram_profile,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)
    log.info("Wrote n-gram metadata to %s", path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(
    language: str,
    top_n: int = DEFAULT_TOP_N,
    input_path: Path | None = None,
    output_dir: Path = MATRICES_DIR,
) -> tuple[dict[str, float], dict[str, float], float]:
    """
    Build bigram and trigram frequency profiles for the given language corpus.

    Expects input JSON with a top-level "sequences" key: a list of token lists.

    Args:
        language:    Language name (used for file naming).
        top_n:       Number of most frequent n-grams to retain.
        input_path:  Path to tokenized corpus JSON.
        output_dir:  Directory for output files.

    Returns:
        (bigram_profile, trigram_profile, type_token_ratio)
    """
    if input_path is None:
        input_path = PROCESSED_DIR / "romance" / f"{language}_tokens.json"

    log.info("Loading tokenized corpus from %s", input_path)
    with input_path.open(encoding="utf-8") as fh:
        corpus = json.load(fh)

    sequences = corpus["sequences"]
    log.info("Loaded %d sequences", len(sequences))

    bigram_counts = extract_ngrams(sequences, 2)
    trigram_counts = extract_ngrams(sequences, 3)
    log.info(
        "Unique bigrams: %d, unique trigrams: %d",
        len(bigram_counts),
        len(trigram_counts),
    )

    bigram_profile = build_profile(bigram_counts, top_n)
    trigram_profile = build_profile(trigram_counts, top_n)
    ttr = type_token_ratio(sequences)
    log.info("Type/token ratio: %.4f", ttr)

    params = {"language": language, "top_n": top_n}
    validate_profile(bigram_profile, "bigram")
    validate_profile(trigram_profile, "trigram")

    meta_path = output_dir / f"{language}_ngram_meta.json"
    write_metadata(bigram_profile, trigram_profile, ttr, params, meta_path)

    return bigram_profile, trigram_profile, ttr


def run_from_sequences(
    language: str,
    sequences: list[list[str]],
    top_n: int = DEFAULT_TOP_N,
    output_dir: Path = MATRICES_DIR,
) -> tuple[dict[str, float], dict[str, float], float]:
    """
    Like run() but accepts sequences directly instead of reading from a JSON file.
    Used by the retrodiction engine to fingerprint generated corpora.
    """
    log.info("Computing n-gram profiles from %d sequences", len(sequences))

    bigram_counts = extract_ngrams(sequences, 2)
    trigram_counts = extract_ngrams(sequences, 3)
    log.info(
        "Unique bigrams: %d, unique trigrams: %d",
        len(bigram_counts),
        len(trigram_counts),
    )

    bigram_profile = build_profile(bigram_counts, top_n)
    trigram_profile = build_profile(trigram_counts, top_n)
    ttr = type_token_ratio(sequences)
    log.info("Type/token ratio: %.4f", ttr)

    params = {"language": language, "top_n": top_n}
    validate_profile(bigram_profile, "bigram")
    validate_profile(trigram_profile, "trigram")

    meta_path = output_dir / f"{language}_ngram_meta.json"
    write_metadata(bigram_profile, trigram_profile, ttr, params, meta_path)

    return bigram_profile, trigram_profile, ttr


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build n-gram frequency profiles")
    parser.add_argument("--language", required=True, help="e.g. french, italian")
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    args = parser.parse_args()
    run(language=args.language, top_n=args.top_n)
