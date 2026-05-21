"""
Positional Frequency Distribution Builder
==========================================
Purpose:
    Build positional frequency distributions from a tokenized language corpus.
    This is one of four statistical fingerprint components used in the
    retrodiction pipeline.

    Positional distributions encode WHERE in a sequence each token tends to appear.
    Initial-position, medial-position, and final-position rates — along with mean
    normalized position — characterize word-order typology and grammatical function.
    Determiners, prepositions, and verbal markers have distinct positional profiles
    that are language-typology signals, independent of what the tokens mean.

    In the retrodiction context, shift in positional distributions is a key signal:
    the movement from analytic (word-order-based) grammar toward synthetic
    (case-based) grammar as iterations proceed backward.

Inputs:
    data/processed/romance/{language}_tokens.json   — tokenized corpus records

Outputs:
    data/matrices/{language}_positional.npy        — positional feature matrix (float32)
    data/matrices/{language}_positional_meta.json  — vocabulary index and parameters

Matrix definition:
    Shape: (V, F) where V = vocabulary size, F = number of position features.

    For each token type i, the feature vector contains:
        [0] initial_rate:     P(position == 0 | token == i)
        [1] final_rate:       P(position == last | token == i)
        [2] medial_rate:      P(0 < position < last | token == i)  (= 1 - initial - final)
        [3] mean_norm_pos:    mean of (position / sequence_length) across all occurrences
        [4] std_norm_pos:     std of normalized positions (spread measure)
        [5] freq_norm:        log-normalized frequency (log(1 + count) / log(1 + max_count))

Assumptions:
    - A token at position 0 is "initial"; at position len-1 is "final"; all others "medial".
    - For single-token sequences, the token is counted as initial only (not final), so
      that initial_rate + final_rate + medial_rate == 1.0 is preserved for all tokens.
    - Sequences are sentences — positional statistics reflect sentence-level word order.

Known limitations:
    - The feature set is fixed at 6 dimensions. If additional positional features are
      needed (e.g. quartile positions), add them and log the change here.
    - Tokens appearing fewer than MIN_OCCURRENCES times have unreliable position
      statistics. They are included but flagged in the metadata.

Validation:
    - Matrix shape must be (V, 6).
    - initial_rate + final_rate + medial_rate == 1.0 for all tokens (within float tolerance).
    - freq_norm values must be in [0.0, 1.0].

Usage:
    python -m src.fingerprint.positional --language italian
    python -m src.fingerprint.positional --language french
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
# Constants
# ---------------------------------------------------------------------------

MIN_OCCURRENCES = 5

F_INITIAL = 0
F_FINAL = 1
F_MEDIAL = 2
F_MEAN_POS = 3
F_STD_POS = 4
F_FREQ_NORM = 5
N_FEATURES = 6


# ---------------------------------------------------------------------------
# Accumulator
# ---------------------------------------------------------------------------

class PositionalAccumulator:
    """
    Accumulates positional statistics for each token type across all sequences.
    Designed for a single pass over the corpus.
    """

    def __init__(self, token2idx: dict[str, int]) -> None:
        V = len(token2idx)
        self.token2idx = token2idx
        self.counts = np.zeros(V, dtype=np.int64)
        self.initial_counts = np.zeros(V, dtype=np.int64)
        self.final_counts = np.zeros(V, dtype=np.int64)
        self.pos_sum = np.zeros(V, dtype=np.float64)
        self.pos_sq_sum = np.zeros(V, dtype=np.float64)

    def ingest_sequence(self, seq: list[str]) -> None:
        n = len(seq)
        if n == 0:
            return

        for i, tok in enumerate(seq):
            if tok not in self.token2idx:
                continue
            idx = self.token2idx[tok]
            norm_pos = i / (n - 1) if n > 1 else 0.0

            self.counts[idx] += 1
            if i == 0:
                self.initial_counts[idx] += 1
            if i == n - 1 and n > 1:
                self.final_counts[idx] += 1
            self.pos_sum[idx] += norm_pos
            self.pos_sq_sum[idx] += norm_pos ** 2

    def to_matrix(self) -> np.ndarray:
        V = len(self.token2idx)
        matrix = np.zeros((V, N_FEATURES), dtype=np.float64)

        safe_counts = np.where(self.counts > 0, self.counts, 1)

        matrix[:, F_INITIAL] = self.initial_counts / safe_counts
        matrix[:, F_FINAL] = self.final_counts / safe_counts
        matrix[:, F_MEDIAL] = 1.0 - matrix[:, F_INITIAL] - matrix[:, F_FINAL]
        matrix[:, F_MEDIAL] = np.clip(matrix[:, F_MEDIAL], 0.0, 1.0)

        matrix[:, F_MEAN_POS] = self.pos_sum / safe_counts

        mean_sq = self.pos_sq_sum / safe_counts
        variance = mean_sq - matrix[:, F_MEAN_POS] ** 2
        variance = np.maximum(variance, 0.0)
        matrix[:, F_STD_POS] = np.sqrt(variance)

        max_count = self.counts.max()
        if max_count > 0:
            matrix[:, F_FREQ_NORM] = np.log1p(self.counts.astype(np.float64)) / np.log1p(max_count)

        matrix[self.counts == 0] = 0.0

        return matrix.astype(np.float32)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_matrix(matrix: np.ndarray, idx2token: list[str]) -> None:
    """Validate the positional feature matrix. Raises ValueError on critical failures."""
    V = len(idx2token)
    if matrix.shape != (V, N_FEATURES):
        raise ValueError(f"Matrix shape {matrix.shape} != ({V}, {N_FEATURES})")

    rate_sum = matrix[:, F_INITIAL] + matrix[:, F_FINAL] + matrix[:, F_MEDIAL]
    nonzero_rows = matrix.sum(axis=1) > 0
    bad = nonzero_rows & ~np.isclose(rate_sum, 1.0, atol=1e-4)
    if bad.any():
        n_bad = bad.sum()
        log.warning("%d tokens have initial+final+medial != 1.0 (max deviation %.6f)",
                    n_bad, np.abs(rate_sum[bad] - 1.0).max())

    freq_out = matrix[:, F_FREQ_NORM]
    if freq_out.min() < -1e-5 or freq_out.max() > 1.0 + 1e-5:
        raise ValueError(f"freq_norm values out of [0, 1]: min={freq_out.min()}, max={freq_out.max()}")

    zero_rows = np.where(~nonzero_rows)[0]
    if len(zero_rows) > 0:
        log.warning("%d tokens have all-zero positional vectors (never appeared in corpus)", len(zero_rows))

    log.info(
        "Positional matrix validation passed: shape=%s, %d zero-rows",
        matrix.shape,
        len(zero_rows),
    )


# ---------------------------------------------------------------------------
# Write outputs
# ---------------------------------------------------------------------------

def write_matrix(matrix: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, matrix)
    log.info("Wrote positional matrix %s to %s", matrix.shape, path)


def write_metadata(
    idx2token: list[str],
    token2idx: dict[str, int],
    counts: np.ndarray,
    params: dict,
    path: Path,
) -> None:
    low_freq_tokens = [
        idx2token[i]
        for i in range(len(idx2token))
        if 0 < counts[i] < MIN_OCCURRENCES
    ]
    meta = {
        "pipeline_phase": "P2",
        "language": params.get("language", "unknown"),
        "n_features": N_FEATURES,
        "feature_names": [
            "initial_rate",
            "final_rate",
            "medial_rate",
            "mean_norm_pos",
            "std_norm_pos",
            "freq_norm",
        ],
        "vocab_size": len(idx2token),
        "idx2token": idx2token,
        "token2idx": token2idx,
        "token_counts": counts.tolist(),
        "low_frequency_tokens": low_freq_tokens,
        "min_occurrences_threshold": MIN_OCCURRENCES,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)
    log.info("Wrote positional metadata to %s", path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(
    language: str,
    input_path: Path | None = None,
    output_dir: Path = MATRICES_DIR,
) -> tuple[np.ndarray, dict[str, int], list[str]]:
    """
    Build positional frequency matrix for the given language corpus.

    Expects input JSON with a top-level "sequences" key: a list of token lists.

    Args:
        language:    Language name (used for file naming).
        input_path:  Path to tokenized corpus JSON.
        output_dir:  Directory for output files.

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

    freq: dict[str, int] = {}
    for seq in sequences:
        for tok in seq:
            freq[tok] = freq.get(tok, 0) + 1
    idx2token = sorted(freq.keys(), key=lambda t: -freq[t])
    token2idx = {tok: i for i, tok in enumerate(idx2token)}
    log.info("Vocabulary: %d unique tokens", len(idx2token))

    accumulator = PositionalAccumulator(token2idx)
    for seq in sequences:
        accumulator.ingest_sequence(seq)

    matrix = accumulator.to_matrix()

    params = {"language": language}
    validate_matrix(matrix, idx2token)

    matrix_path = output_dir / f"{language}_positional.npy"
    meta_path = output_dir / f"{language}_positional_meta.json"
    write_matrix(matrix, matrix_path)
    write_metadata(idx2token, token2idx, accumulator.counts, params, meta_path)

    return matrix, token2idx, idx2token


def run_from_sequences(
    language: str,
    sequences: list[list[str]],
    output_dir: Path = MATRICES_DIR,
) -> tuple[np.ndarray, dict[str, int], list[str]]:
    """
    Like run() but accepts sequences directly instead of reading from a JSON file.
    Used by the retrodiction engine to fingerprint generated corpora.
    """
    freq: dict[str, int] = {}
    for seq in sequences:
        for tok in seq:
            freq[tok] = freq.get(tok, 0) + 1
    idx2token = sorted(freq.keys(), key=lambda t: -freq[t])
    token2idx = {tok: i for i, tok in enumerate(idx2token)}
    log.info("Vocabulary: %d unique tokens", len(idx2token))

    accumulator = PositionalAccumulator(token2idx)
    for seq in sequences:
        accumulator.ingest_sequence(seq)

    matrix = accumulator.to_matrix()

    params = {"language": language}
    validate_matrix(matrix, idx2token)

    matrix_path = output_dir / f"{language}_positional.npy"
    meta_path = output_dir / f"{language}_positional_meta.json"
    write_matrix(matrix, matrix_path)
    write_metadata(idx2token, token2idx, accumulator.counts, params, meta_path)

    return matrix, token2idx, idx2token


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build positional frequency distribution")
    parser.add_argument("--language", required=True, help="e.g. french, italian")
    args = parser.parse_args()
    run(language=args.language)
