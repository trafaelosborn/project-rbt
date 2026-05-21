"""
Unicode Text Tokenizer
======================
Purpose:
    Convert raw natural language text (Wikipedia article prose in Romance languages
    or Latin) into normalized token sequences suitable for statistical fingerprinting.
    Tokens are treated as abstract units — no linguistic assumptions are applied
    beyond the mechanics of word segmentation.

Inputs:
    A string of raw text (typically one Wikipedia article extract or Latin passage).

Outputs:
    list[list[str]] — one inner list per sentence; each inner list is a sequence
    of normalized word tokens.

Token model:
    The fundamental unit is the WORD TOKEN, defined as a maximal sequence of
    Unicode alphabetic characters (no digits, no punctuation, no whitespace).
    Contractions (French "l'eau", "c'est") are split at the apostrophe, yielding
    two tokens ("l", "eau"). This is consistent across all languages and introduces
    no language-specific assumptions.

    Token length filter: tokens shorter than MIN_TOKEN_LEN characters are dropped.
    This removes single-letter residues from contraction splitting and stray
    characters from encoding artifacts.

Normalization rules:
    1. Unicode NFC normalization (canonical composition).
    2. Lowercase.
    3. Extract sequences of Unicode letters only (regex \\p{L}+, via [^\\W\\d_]+).
    4. Drop tokens shorter than MIN_TOKEN_LEN (default: 2).
    5. Drop tokens matching the stop-pattern list (numerals written as words are
       kept — they are valid tokens in statistical analysis).

Sentence splitting:
    Sentences are split on ". ", "! ", "? " followed by an uppercase letter, or
    on newlines. This is intentionally simple and language-agnostic. It will
    mis-split on abbreviations (e.g. "Dr. Smith") — this is an acceptable noise
    source given the corpus size. A more sophisticated sentence splitter would
    require language-specific models, which would introduce implicit linguistic
    assumptions.

    Decision log: docs/decisions/004_sentence_splitting.md

Assumptions:
    - Input text is UTF-8 encoded Unicode prose.
    - Wikipedia extracts are reasonably clean — minimal HTML artifacts after API
      extraction.
    - All languages are processed identically; no language-specific rules.

Known limitations:
    - Contraction splitting ("l'" → "l") produces a high-frequency single-character
      token "l" in French. MIN_TOKEN_LEN=2 drops it. This removes the French elision
      article from the token vocabulary, which is a meaningful loss. Documented here
      as a deliberate trade-off for cross-language consistency.
    - Abbreviation-induced false sentence splits add short sequences to the corpus.
      These contribute noise but are a small fraction of total sequences.
    - Genoese and Occitan Wikipedia prose may contain code-switching with Italian
      or French. Not filtered; flagged in corpus_sources.md.

Usage:
    from src.ingest.tokenize import tokenize_text, tokenize_corpus

    sequences = tokenize_text("Il presidente ha dichiarato che...")
    # → [["il", "presidente", "ha", "dichiarato", "che", ...], ...]
"""

import logging
import re
import unicodedata
from typing import Iterator

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Tokens shorter than this are dropped.
MIN_TOKEN_LEN = 2

# Regex matching a maximal run of Unicode alphabetic characters.
# [^\W\d_]+ matches Unicode letters only (no digits, underscores, punctuation).
_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)

# Sentence boundary: period/exclamation/question followed by whitespace + uppercase,
# or a newline. We split on the boundary and keep the punctuation with the left side.
_SENT_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+(?=[A-ZÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖØÙÚÛÜÝÞ])|[\n\r]+")


# ---------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------

def split_sentences(text: str) -> list[str]:
    """
    Split text into sentence strings using simple punctuation heuristics.
    No language-specific models are used.

    Returns a list of non-empty sentence strings.
    """
    parts = _SENT_BOUNDARY_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


# ---------------------------------------------------------------------------
# Word tokenization
# ---------------------------------------------------------------------------

def tokenize_sentence(sentence: str) -> list[str]:
    """
    Tokenize a single sentence string into word tokens.

    Steps:
        1. NFC normalization.
        2. Extract Unicode letter runs.
        3. Lowercase.
        4. Drop tokens shorter than MIN_TOKEN_LEN.

    Returns a (possibly empty) list of token strings.
    """
    sentence = unicodedata.normalize("NFC", sentence)
    tokens = _WORD_RE.findall(sentence)
    return [tok.lower() for tok in tokens if len(tok) >= MIN_TOKEN_LEN]


# ---------------------------------------------------------------------------
# Full text tokenization
# ---------------------------------------------------------------------------

def tokenize_text(text: str) -> list[list[str]]:
    """
    Tokenize a full text into sentence-level token sequences.

    Args:
        text: Raw Unicode prose.

    Returns:
        List of token sequences (one per sentence). Empty sentences are excluded.
    """
    sentences = split_sentences(text)
    sequences = []
    for sent in sentences:
        tokens = tokenize_sentence(sent)
        if tokens:
            sequences.append(tokens)
    return sequences


# ---------------------------------------------------------------------------
# Corpus-level helpers
# ---------------------------------------------------------------------------

def tokenize_corpus(
    texts: Iterator[str],
    log_interval: int = 100,
) -> list[list[str]]:
    """
    Tokenize an iterable of text strings into a flat list of token sequences.
    Each text is split into sentences independently (no sequences span text boundaries).

    Args:
        texts:        Iterable of raw text strings (e.g. Wikipedia article extracts).
        log_interval: Log progress every N texts.

    Returns:
        All sequences from all texts, concatenated.
    """
    all_sequences: list[list[str]] = []
    for i, text in enumerate(texts):
        sequences = tokenize_text(text)
        all_sequences.extend(sequences)
        if (i + 1) % log_interval == 0:
            log.info(
                "Tokenized %d texts → %d sequences so far",
                i + 1,
                len(all_sequences),
            )
    return all_sequences


def corpus_stats(sequences: list[list[str]]) -> dict:
    """
    Compute summary statistics for a tokenized corpus.

    Returns a dict with:
        sequence_count:  number of sequences (sentences)
        total_tokens:    total token count across all sequences
        unique_types:    number of unique token types
        type_token_ratio: unique_types / total_tokens
        mean_seq_length: average tokens per sequence
    """
    total = sum(len(s) for s in sequences)
    types: set[str] = set()
    for seq in sequences:
        types.update(seq)
    return {
        "sequence_count": len(sequences),
        "total_tokens": total,
        "unique_types": len(types),
        "type_token_ratio": len(types) / total if total > 0 else 0.0,
        "mean_seq_length": total / len(sequences) if sequences else 0.0,
    }
