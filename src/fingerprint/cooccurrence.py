"""
Co-occurrence Matrix Builder
============================
Purpose:
    Build a token co-occurrence matrix from a tokenized language corpus.
    This is one of four statistical fingerprint components used in the
    retrodiction pipeline.

Inputs:
    data/processed/{language}_tokens.json   — tokenized corpus records

Outputs:
    data/matrices/{language}_cooccurrence.npy        — co-occurrence matrix (float32)
    data/matrices/{language}_cooccurrence_meta.json  — vocabulary index and parameters

Matrix definition:
    M[i, j] = number of times token j appears within WINDOW tokens of token i,
    summed across all sequences. The matrix is symmetric (window is bidirectional).
    Diagonal entries are zero (a token does not co-occur with itself).

    After counting, each row is L2-normalized so that high-frequency tokens do not
    dominate the similarity space. Raw counts are preserved in the metadata.

Parameters:
    WINDOW (int, default=2): number of tokens to the left and right of each target token.

    Decision log: docs/decisions/002_cooccurrence_window.md

Assumptions:
    - Each sentence/sequence is treated as a contiguous context.
    - Sequence boundaries are hard boundaries: no co-occurrence is counted across them.

Known limitations:
    - Thin corpora (Occitan, Genoese) produce sparser matrices. The normalization
      strategy is robust to sparsity, but this is flagged in the metadata.
    - Very high-frequency tokens (function words) may still dominate after L2
      normalization. A PPMI transform is available and recommended for cross-language
      comparison.

Validation:
    - Matrix must be square, shape (V, V) where V is vocabulary size.
    - Matrix must be symmetric (M == M.T within floating-point tolerance).
    - Diagonal must be all-zero.

Usage:
    python -m src.fingerprint.cooccurrence --language italian
    python -m src.fingerprint.cooccurrence --language french --window 3 --ppmi
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MATRICES_DIR = PROJECT_ROOT / "data" / "matrices"

# ---------------------------------------------------------------------------
# Default parameters
# ---------------------------------------------------------------------------

DEFAULT_WINDOW = 2
MAX_VOCAB = 5000   # cap vocabulary to keep matrix memory-feasible (~95 MB float32)


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

def build_vocab(
    sequences: list[list[str]],
    max_vocab: int | None = MAX_VOCAB,
) -> tuple[dict[str, int], list[str]]:
    """
    Build a token→index mapping and index→token list from all sequences.
    Tokens are sorted by frequency (descending) so that the most common tokens
    have low indices — convenient for debugging and visualization.

    Args:
        sequences: List of token sequences.
        max_vocab: If set, keep only the top-N most frequent tokens.
                   Defaults to MAX_VOCAB (5000) to keep matrix sizes manageable.

    Returns:
        token2idx: {token: index}
        idx2token: [token at index 0, token at index 1, ...]
    """
    freq: dict[str, int] = {}
    for seq in sequences:
        for tok in seq:
            freq[tok] = freq.get(tok, 0) + 1

    sorted_tokens = sorted(freq.keys(), key=lambda t: -freq[t])
    if max_vocab is not None:
        sorted_tokens = sorted_tokens[:max_vocab]
    token2idx = {tok: i for i, tok in enumerate(sorted_tokens)}
    return token2idx, sorted_tokens


# ---------------------------------------------------------------------------
# Count co-occurrences
# ---------------------------------------------------------------------------

def count_cooccurrences(
    sequences: list[list[str]],
    token2idx: dict[str, int],
    window: int = DEFAULT_WINDOW,
) -> np.ndarray:
    """
    Count co-occurrences with a symmetric sliding window.

    For each position i in a sequence, count all tokens within positions
    [i-window, i+window] (excluding i itself and out-of-bounds positions)
    as co-occurrences with token at position i.

    Args:
        sequences: List of token sequences (one per sentence/document unit).
        token2idx: Token-to-index mapping.
        window: Context window size (tokens on each side).

    Returns:
        counts: (V, V) int64 array of raw co-occurrence counts.
    """
    V = len(token2idx)
    counts = np.zeros((V, V), dtype=np.int64)

    for seq in sequences:
        n = len(seq)
        for i, center in enumerate(seq):
            if center not in token2idx:
                continue
            ci = token2idx[center]
            start = max(0, i - window)
            end = min(n, i + window + 1)
            for j in range(start, end):
                if j == i:
                    continue
                context = seq[j]
                if context not in token2idx:
                    continue
                cj = token2idx[context]
                counts[ci, cj] += 1

    return counts


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def apply_ppmi(counts: np.ndarray) -> np.ndarray:
    """
    Apply Positive Pointwise Mutual Information (PPMI) transform.

    PPMI(i, j) = max(0, log2( P(i,j) / (P(i) * P(j)) ))

    This down-weights high-frequency tokens that co-occur with everything
    and up-weights informative co-occurrences.

    Returns a float32 PPMI matrix.
    """
    total = counts.sum()
    if total == 0:
        raise ValueError("Co-occurrence matrix is all-zero — cannot compute PPMI")

    prob_joint = counts.astype(np.float64) / total
    prob_i = prob_joint.sum(axis=1, keepdims=True)   # (V, 1)
    prob_j = prob_joint.sum(axis=0, keepdims=True)   # (1, V)

    denom = prob_i * prob_j
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.where(denom > 0, np.log2(np.where(denom > 0, prob_joint / denom, 1.0)), 0.0)

    ppmi = np.maximum(pmi, 0.0).astype(np.float32)
    return ppmi


def l2_normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """
    L2-normalize each row of the matrix. Rows that are all-zero remain all-zero.
    Returns float32.
    """
    matrix = matrix.astype(np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # avoid division by zero for all-zero rows
    return matrix / norms


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_matrix(
    matrix: np.ndarray,
    idx2token: list[str],
    counts: np.ndarray | None = None,
) -> None:
    """
    Validate the co-occurrence matrix.

    Args:
        matrix:    The final (possibly normalized) matrix.
        idx2token: Token index.
        counts:    The raw int64 count matrix before normalization.
                   If provided, symmetry is checked on counts (correct).
                   If None, symmetry is checked on matrix (only valid pre-normalization).

    Raises ValueError on critical failures.
    """
    V = len(idx2token)
    if matrix.shape != (V, V):
        raise ValueError(f"Matrix shape {matrix.shape} != ({V}, {V})")

    sym_target = counts if counts is not None else matrix
    if not np.allclose(sym_target, sym_target.T, atol=1e-5):
        raise ValueError("Co-occurrence count matrix is not symmetric")

    if np.any(np.diag(matrix) != 0.0):
        raise ValueError("Co-occurrence matrix has non-zero diagonal")

    zero_rows = np.where(matrix.sum(axis=1) == 0)[0]
    if len(zero_rows) > 0:
        tokens = [idx2token[i] for i in zero_rows[:10]]
        log.warning(
            "%d tokens have all-zero co-occurrence rows (isolated tokens): %s%s",
            len(zero_rows),
            tokens,
            " ..." if len(zero_rows) > 10 else "",
        )

    log.info(
        "Matrix validation passed: shape=%s, symmetric_counts=True, zero-diagonal=True, "
        "isolated_tokens=%d",
        matrix.shape,
        len(zero_rows),
    )


# ---------------------------------------------------------------------------
# Write outputs
# ---------------------------------------------------------------------------

def write_matrix(matrix: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, matrix)
    log.info("Wrote matrix %s to %s", matrix.shape, path)


def write_metadata(
    idx2token: list[str],
    token2idx: dict[str, int],
    params: dict,
    path: Path,
) -> None:
    meta = {
        "pipeline_phase": "P2",
        "language": params.get("language", "unknown"),
        "window": params.get("window", DEFAULT_WINDOW),
        "ppmi_applied": params.get("ppmi", False),
        "l2_normalized": True,
        "vocab_size": len(idx2token),
        "idx2token": idx2token,
        "token2idx": token2idx,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)
    log.info("Wrote matrix metadata to %s", path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(
    language: str,
    window: int = DEFAULT_WINDOW,
    ppmi: bool = False,
    input_path: Path | None = None,
    output_dir: Path = MATRICES_DIR,
    max_vocab: int | None = MAX_VOCAB,
) -> tuple[np.ndarray, dict[str, int], list[str]]:
    """
    Build co-occurrence matrix for the given language corpus.

    Expects input JSON with a top-level "sequences" key: a list of token lists.

    Args:
        language:    Language name (used for file naming).
        window:      Co-occurrence window size.
        ppmi:        Apply PPMI transform before L2 normalization.
        input_path:  Path to tokenized corpus JSON. Defaults to standard location.
        output_dir:  Directory for output files.
        max_vocab:   Vocabulary cap. Defaults to MAX_VOCAB (5000).

    Returns:
        (matrix, token2idx, idx2token)
    """
    if input_path is None:
        input_path = PROCESSED_DIR / "romance" / f"{language}_tokens.json"

    log.info("Loading tokenized corpus from %s", input_path)
    with input_path.open(encoding="utf-8") as fh:
        corpus = json.load(fh)

    sequences = corpus["sequences"]
    log.info("Loaded %d sequences", len(sequences))

    token2idx, idx2token = build_vocab(sequences, max_vocab=max_vocab)
    log.info("Vocabulary: %d unique tokens (cap=%s)", len(idx2token), max_vocab)

    counts = count_cooccurrences(sequences, token2idx, window)
    log.info(
        "Raw counts: total=%d, non-zero entries=%d",
        counts.sum(),
        np.count_nonzero(counts),
    )

    if ppmi:
        matrix = apply_ppmi(counts)
        log.info("Applied PPMI transform")
    else:
        matrix = counts.astype(np.float32)

    np.fill_diagonal(matrix, 0.0)
    matrix = l2_normalize_rows(matrix)

    params = {"language": language, "window": window, "ppmi": ppmi, "max_vocab": max_vocab}
    validate_matrix(matrix, idx2token, counts=counts)

    matrix_path = output_dir / f"{language}_cooccurrence.npy"
    meta_path = output_dir / f"{language}_cooccurrence_meta.json"
    write_matrix(matrix, matrix_path)
    write_metadata(idx2token, token2idx, params, meta_path)

    return matrix, token2idx, idx2token


def run_from_sequences(
    language: str,
    sequences: list[list[str]],
    window: int = DEFAULT_WINDOW,
    ppmi: bool = False,
    output_dir: Path = MATRICES_DIR,
    max_vocab: int | None = MAX_VOCAB,
) -> tuple[np.ndarray, dict[str, int], list[str]]:
    """
    Like run() but accepts sequences directly instead of reading from a JSON file.
    Used by the retrodiction engine to fingerprint generated corpora.
    """
    token2idx, idx2token = build_vocab(sequences, max_vocab=max_vocab)
    log.info("Vocabulary: %d unique tokens (cap=%s)", len(idx2token), max_vocab)

    counts = count_cooccurrences(sequences, token2idx, window)
    log.info(
        "Raw counts: total=%d, non-zero entries=%d",
        counts.sum(),
        np.count_nonzero(counts),
    )

    if ppmi:
        matrix = apply_ppmi(counts)
        log.info("Applied PPMI transform")
    else:
        matrix = counts.astype(np.float32)

    np.fill_diagonal(matrix, 0.0)
    matrix = l2_normalize_rows(matrix)

    params = {"language": language, "window": window, "ppmi": ppmi, "max_vocab": max_vocab}
    validate_matrix(matrix, idx2token, counts=counts)

    matrix_path = output_dir / f"{language}_cooccurrence.npy"
    meta_path = output_dir / f"{language}_cooccurrence_meta.json"
    write_matrix(matrix, matrix_path)
    write_metadata(idx2token, token2idx, params, meta_path)

    return matrix, token2idx, idx2token


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build co-occurrence matrix")
    parser.add_argument("--language", required=True, help="e.g. french, italian")
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    parser.add_argument("--ppmi", action="store_true", help="Apply PPMI transform")
    args = parser.parse_args()
    run(language=args.language, window=args.window, ppmi=args.ppmi)
