"""
Phase 4 incremental fingerprint tensor state.

This module is the first scaffold toward a delta-native acceleration path:

- keep a live tensor state in memory
- update accepted mutations against that state
- fall back to a full rebuild when vocabulary drift breaks the current anchor

It does not yet replace the existing engine loop. It provides the state object
and update semantics needed for that later step.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.accelerate.tensor_layout import FingerprintTensorLayout
from src.fingerprint.cooccurrence import (
    DEFAULT_WINDOW,
    MAX_VOCAB,
    build_vocab,
    count_cooccurrences,
    l2_normalize_rows,
)
from src.fingerprint.ngram import DEFAULT_TOP_N, extract_ngrams
from src.fingerprint.positional import N_FEATURES, PositionalAccumulator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MATRICES_DIR = PROJECT_ROOT / "data" / "matrices"


@dataclass(frozen=True)
class TensorStateConfig:
    max_vocab: int = MAX_VOCAB
    cooccurrence_window: int = DEFAULT_WINDOW
    ngram_top_n: int = DEFAULT_TOP_N

    def to_dict(self) -> dict:
        return {
            "max_vocab": self.max_vocab,
            "cooccurrence_window": self.cooccurrence_window,
            "ngram_top_n": self.ngram_top_n,
        }


@dataclass(frozen=True)
class NgramVectorBasis:
    bigram_keys: tuple[str, ...]
    trigram_keys: tuple[str, ...]

    @classmethod
    def from_ngram_meta_path(cls, path: Path) -> "NgramVectorBasis":
        with path.open(encoding="utf-8") as fh:
            meta = json.load(fh)
        return cls(
            bigram_keys=tuple(meta.get("bigrams", {}).keys()),
            trigram_keys=tuple(meta.get("trigrams", {}).keys()),
        )

    @classmethod
    def from_reference_label(
        cls,
        reference_label: str = "latin",
        *,
        matrices_dir: Path = MATRICES_DIR,
    ) -> "NgramVectorBasis":
        return cls.from_ngram_meta_path(matrices_dir / f"{reference_label}_ngram_meta.json")

    def vectorize_profile(self, profile: dict[str, float], kind: str) -> np.ndarray:
        if kind == "bigram":
            keys = self.bigram_keys
        elif kind == "trigram":
            keys = self.trigram_keys
        else:
            raise ValueError(f"kind must be 'bigram' or 'trigram', got {kind!r}")
        return np.asfortranarray([float(profile.get(key, 0.0)) for key in keys], dtype=np.float64)

    def manifest(self) -> dict:
        return {
            "bigram_size": len(self.bigram_keys),
            "trigram_size": len(self.trigram_keys),
        }


@dataclass(frozen=True)
class SequenceDelta:
    changed_indices: tuple[int, ...]
    added_tokens: tuple[str, ...]
    removed_tokens: tuple[str, ...]
    touched_tokens: tuple[str, ...]
    touched_bigrams: tuple[str, ...]
    touched_trigrams: tuple[str, ...]
    sequence_count_changed: bool

    @property
    def changed_sequence_count(self) -> int:
        return len(self.changed_indices)

    @property
    def is_noop(self) -> bool:
        return self.changed_sequence_count == 0 and not self.sequence_count_changed

    def to_dict(self) -> dict:
        return {
            "changed_indices": list(self.changed_indices),
            "changed_sequence_count": self.changed_sequence_count,
            "added_tokens": list(self.added_tokens),
            "removed_tokens": list(self.removed_tokens),
            "touched_tokens": list(self.touched_tokens),
            "touched_bigrams": list(self.touched_bigrams),
            "touched_trigrams": list(self.touched_trigrams),
            "sequence_count_changed": self.sequence_count_changed,
        }


@dataclass(frozen=True)
class TensorUpdateResult:
    mode: str
    delta: SequenceDelta
    anchor_vocab_size: int
    oov_tokens: tuple[str, ...]
    bigram_vector_size: int
    trigram_vector_size: int

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "delta": self.delta.to_dict(),
            "anchor_vocab_size": self.anchor_vocab_size,
            "oov_tokens": list(self.oov_tokens),
            "bigram_vector_size": self.bigram_vector_size,
            "trigram_vector_size": self.trigram_vector_size,
        }


def _clone_sequences(sequences: list[list[str]]) -> list[list[str]]:
    return [list(seq) for seq in sequences]


def _sequence_ngrams(seq: list[str], n: int) -> Counter:
    counter: Counter = Counter()
    for i in range(len(seq) - n + 1):
        gram = tuple(seq[i : i + n])
        counter[gram] += 1
    return counter


def _sequence_cooccurrence_update(
    counts: np.ndarray,
    seq: list[str],
    token2idx: dict[str, int],
    *,
    window: int,
    sign: int,
) -> None:
    n = len(seq)
    for i, center in enumerate(seq):
        ci = token2idx.get(center)
        if ci is None:
            continue
        start = max(0, i - window)
        end = min(n, i + window + 1)
        for j in range(start, end):
            if j == i:
                continue
            cj = token2idx.get(seq[j])
            if cj is None:
                continue
            counts[ci, cj] += sign


def _sequence_positional_update(
    counts: np.ndarray,
    initial_counts: np.ndarray,
    final_counts: np.ndarray,
    pos_sum: np.ndarray,
    pos_sq_sum: np.ndarray,
    seq: list[str],
    token2idx: dict[str, int],
    *,
    sign: int,
) -> None:
    n = len(seq)
    if n == 0:
        return
    for i, tok in enumerate(seq):
        idx = token2idx.get(tok)
        if idx is None:
            continue
        norm_pos = i / (n - 1) if n > 1 else 0.0
        counts[idx] += sign
        if i == 0:
            initial_counts[idx] += sign
        if i == n - 1 and n > 1:
            final_counts[idx] += sign
        pos_sum[idx] += sign * norm_pos
        pos_sq_sum[idx] += sign * (norm_pos ** 2)


def _build_positional_matrix(
    counts: np.ndarray,
    initial_counts: np.ndarray,
    final_counts: np.ndarray,
    pos_sum: np.ndarray,
    pos_sq_sum: np.ndarray,
) -> np.ndarray:
    safe_counts = np.where(counts > 0, counts, 1)
    matrix = np.zeros((counts.size, N_FEATURES), dtype=np.float64, order="F")
    matrix[:, 0] = initial_counts / safe_counts
    matrix[:, 1] = final_counts / safe_counts
    matrix[:, 2] = 1.0 - matrix[:, 0] - matrix[:, 1]
    matrix[:, 2] = np.clip(matrix[:, 2], 0.0, 1.0)
    matrix[:, 3] = pos_sum / safe_counts
    mean_sq = pos_sq_sum / safe_counts
    variance = np.maximum(mean_sq - matrix[:, 3] ** 2, 0.0)
    matrix[:, 4] = np.sqrt(variance)

    max_count = int(counts.max(initial=0))
    if max_count > 0:
        matrix[:, 5] = np.log1p(counts.astype(np.float64)) / np.log1p(max_count)
    matrix[counts == 0] = 0.0
    return np.asfortranarray(matrix, dtype=np.float64)


def _count_tokens(sequences: list[list[str]]) -> Counter:
    return Counter(tok for seq in sequences for tok in seq)


def _build_deterministic_profile(counter: Counter, top_n: int) -> dict[str, float]:
    if top_n <= 0:
        return {}

    ordered = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    top = [(gram, count) for gram, count in ordered[:top_n] if count > 0]
    total = sum(count for _, count in top)
    if total == 0:
        return {}
    return {" | ".join(gram): count / total for gram, count in top}


def compute_sequence_delta(
    before_sequences: list[list[str]],
    after_sequences: list[list[str]],
) -> SequenceDelta:
    max_len = max(len(before_sequences), len(after_sequences))
    changed_indices: list[int] = []
    old_token_counts: Counter = Counter()
    new_token_counts: Counter = Counter()
    touched_bigrams: Counter = Counter()
    touched_trigrams: Counter = Counter()

    for idx in range(max_len):
        before = before_sequences[idx] if idx < len(before_sequences) else []
        after = after_sequences[idx] if idx < len(after_sequences) else []
        if before == after:
            continue
        changed_indices.append(idx)
        old_token_counts.update(before)
        new_token_counts.update(after)
        touched_bigrams.update(_sequence_ngrams(before, 2))
        touched_bigrams.update(_sequence_ngrams(after, 2))
        touched_trigrams.update(_sequence_ngrams(before, 3))
        touched_trigrams.update(_sequence_ngrams(after, 3))

    added_tokens = sorted(tok for tok, count in (new_token_counts - old_token_counts).items() if count > 0)
    removed_tokens = sorted(tok for tok, count in (old_token_counts - new_token_counts).items() if count > 0)
    touched_tokens = sorted(set(old_token_counts) | set(new_token_counts))

    return SequenceDelta(
        changed_indices=tuple(changed_indices),
        added_tokens=tuple(added_tokens),
        removed_tokens=tuple(removed_tokens),
        touched_tokens=tuple(touched_tokens),
        touched_bigrams=tuple(sorted(" | ".join(gram) for gram in touched_bigrams.keys())),
        touched_trigrams=tuple(sorted(" | ".join(gram) for gram in touched_trigrams.keys())),
        sequence_count_changed=(len(before_sequences) != len(after_sequences)),
    )


class IncrementalFingerprintTensorState:
    """
    Live fingerprint tensor state for a bridge corpus.

    The object can:
    - build the packed fingerprint tensor from sequences
    - apply accepted-mutation updates incrementally when the anchor vocabulary survives
    - fall back to a full rebuild when new OOV tokens force re-anchoring
    """

    def __init__(
        self,
        *,
        sequences: list[list[str]],
        config: TensorStateConfig,
        ngram_basis: NgramVectorBasis | None,
        idx2token: list[str],
        token2idx: dict[str, int],
        token_counts: Counter,
        cooccurrence_counts: np.ndarray,
        positional_counts: np.ndarray,
        positional_initial_counts: np.ndarray,
        positional_final_counts: np.ndarray,
        positional_pos_sum: np.ndarray,
        positional_pos_sq_sum: np.ndarray,
        bigram_counts: Counter,
        trigram_counts: Counter,
    ) -> None:
        self.config = config
        self.ngram_basis = ngram_basis
        self.sequences = _clone_sequences(sequences)
        self.idx2token = list(idx2token)
        self.token2idx = dict(token2idx)
        self.token_counts = Counter(token_counts)
        self.cooccurrence_counts = np.asarray(cooccurrence_counts, dtype=np.int64, order="F")
        self.positional_counts = np.asarray(positional_counts, dtype=np.int64)
        self.positional_initial_counts = np.asarray(positional_initial_counts, dtype=np.int64)
        self.positional_final_counts = np.asarray(positional_final_counts, dtype=np.int64)
        self.positional_pos_sum = np.asarray(positional_pos_sum, dtype=np.float64)
        self.positional_pos_sq_sum = np.asarray(positional_pos_sq_sum, dtype=np.float64)
        self.bigram_counts = Counter(bigram_counts)
        self.trigram_counts = Counter(trigram_counts)

        self.cooccurrence_matrix = np.zeros((len(self.idx2token), len(self.idx2token)), dtype=np.float64, order="F")
        self.positional_matrix = np.zeros((len(self.idx2token), N_FEATURES), dtype=np.float64, order="F")
        self.bigram_profile: dict[str, float] = {}
        self.trigram_profile: dict[str, float] = {}
        self.bigram_vector = np.zeros(0, dtype=np.float64)
        self.trigram_vector = np.zeros(0, dtype=np.float64)
        self.layout = FingerprintTensorLayout(
            vocab_size=max(len(self.idx2token), 1),
            positional_width=N_FEATURES,
            bigram_profile_size=0,
            trigram_profile_size=0,
        )
        self.tensor = np.zeros(self.layout.total_size, dtype=np.float64)
        self._refresh_views()

    @classmethod
    def from_sequences(
        cls,
        sequences: list[list[str]],
        *,
        config: TensorStateConfig | None = None,
        ngram_basis: NgramVectorBasis | None = None,
    ) -> "IncrementalFingerprintTensorState":
        cfg = config or TensorStateConfig()
        token2idx, idx2token = build_vocab(sequences, max_vocab=cfg.max_vocab)
        token_counts = _count_tokens(sequences)

        cooccurrence_counts = count_cooccurrences(
            sequences,
            token2idx,
            window=cfg.cooccurrence_window,
        )

        positional_accumulator = PositionalAccumulator(token2idx)
        for seq in sequences:
            positional_accumulator.ingest_sequence(seq)

        bigram_counts = extract_ngrams(sequences, 2)
        trigram_counts = extract_ngrams(sequences, 3)

        return cls(
            sequences=sequences,
            config=cfg,
            ngram_basis=ngram_basis,
            idx2token=idx2token,
            token2idx=token2idx,
            token_counts=token_counts,
            cooccurrence_counts=cooccurrence_counts,
            positional_counts=positional_accumulator.counts,
            positional_initial_counts=positional_accumulator.initial_counts,
            positional_final_counts=positional_accumulator.final_counts,
            positional_pos_sum=positional_accumulator.pos_sum,
            positional_pos_sq_sum=positional_accumulator.pos_sq_sum,
            bigram_counts=bigram_counts,
            trigram_counts=trigram_counts,
        )

    @classmethod
    def from_sequences_with_anchor(
        cls,
        sequences: list[list[str]],
        *,
        anchor_tokens: list[str],
        config: TensorStateConfig | None = None,
        ngram_basis: NgramVectorBasis | None = None,
    ) -> "IncrementalFingerprintTensorState":
        cfg = config or TensorStateConfig()
        idx2token = list(anchor_tokens)
        token2idx = {tok: i for i, tok in enumerate(idx2token)}
        token_counts = _count_tokens(sequences)

        cooccurrence_counts = count_cooccurrences(
            sequences,
            token2idx,
            window=cfg.cooccurrence_window,
        )

        positional_accumulator = PositionalAccumulator(token2idx)
        for seq in sequences:
            positional_accumulator.ingest_sequence(seq)

        bigram_counts = extract_ngrams(sequences, 2)
        trigram_counts = extract_ngrams(sequences, 3)

        return cls(
            sequences=sequences,
            config=cfg,
            ngram_basis=ngram_basis,
            idx2token=idx2token,
            token2idx=token2idx,
            token_counts=token_counts,
            cooccurrence_counts=cooccurrence_counts,
            positional_counts=positional_accumulator.counts,
            positional_initial_counts=positional_accumulator.initial_counts,
            positional_final_counts=positional_accumulator.final_counts,
            positional_pos_sum=positional_accumulator.pos_sum,
            positional_pos_sq_sum=positional_accumulator.pos_sq_sum,
            bigram_counts=bigram_counts,
            trigram_counts=trigram_counts,
        )

    def _refresh_views(self) -> None:
        self.cooccurrence_matrix = np.asfortranarray(
            l2_normalize_rows(self.cooccurrence_counts),
            dtype=np.float64,
        )
        self.positional_matrix = _build_positional_matrix(
            self.positional_counts,
            self.positional_initial_counts,
            self.positional_final_counts,
            self.positional_pos_sum,
            self.positional_pos_sq_sum,
        )
        self.bigram_profile = _build_deterministic_profile(self.bigram_counts, self.config.ngram_top_n)
        self.trigram_profile = _build_deterministic_profile(self.trigram_counts, self.config.ngram_top_n)

        if self.ngram_basis is None:
            self.bigram_vector = np.zeros(0, dtype=np.float64)
            self.trigram_vector = np.zeros(0, dtype=np.float64)
        else:
            self.bigram_vector = self.ngram_basis.vectorize_profile(self.bigram_profile, "bigram")
            self.trigram_vector = self.ngram_basis.vectorize_profile(self.trigram_profile, "trigram")

        self.layout = FingerprintTensorLayout(
            vocab_size=max(len(self.idx2token), 1),
            positional_width=N_FEATURES,
            bigram_profile_size=self.bigram_vector.size,
            trigram_profile_size=self.trigram_vector.size,
        )
        self.tensor = self.layout.pack(
            cooccurrence=self.cooccurrence_matrix,
            positional=self.positional_matrix,
            bigram_profile=self.bigram_vector,
            trigram_profile=self.trigram_vector,
        )

    def manifest(self) -> dict:
        return {
            "config": self.config.to_dict(),
            "anchor_vocab_size": len(self.idx2token),
            "sequence_count": len(self.sequences),
            "ngram_basis": None if self.ngram_basis is None else self.ngram_basis.manifest(),
            "layout": self.layout.manifest(),
        }

    def _replace_with(self, rebuilt: "IncrementalFingerprintTensorState") -> None:
        self.__dict__.update(rebuilt.__dict__)

    def _extend_anchor(self, new_tokens: tuple[str, ...]) -> None:
        extension = [tok for tok in new_tokens if tok not in self.token2idx]
        if not extension:
            return

        old_size = len(self.idx2token)
        new_size = old_size + len(extension)

        expanded_cooccurrence = np.zeros((new_size, new_size), dtype=np.int64, order="F")
        expanded_cooccurrence[:old_size, :old_size] = self.cooccurrence_counts
        self.cooccurrence_counts = expanded_cooccurrence

        def _extend_vector(vec: np.ndarray, *, dtype) -> np.ndarray:
            expanded = np.zeros(new_size, dtype=dtype)
            expanded[:old_size] = vec
            return expanded

        self.positional_counts = _extend_vector(self.positional_counts, dtype=np.int64)
        self.positional_initial_counts = _extend_vector(self.positional_initial_counts, dtype=np.int64)
        self.positional_final_counts = _extend_vector(self.positional_final_counts, dtype=np.int64)
        self.positional_pos_sum = _extend_vector(self.positional_pos_sum, dtype=np.float64)
        self.positional_pos_sq_sum = _extend_vector(self.positional_pos_sq_sum, dtype=np.float64)

        for tok in extension:
            idx = len(self.idx2token)
            self.idx2token.append(tok)
            self.token2idx[tok] = idx

    def apply_sequences(self, new_sequences: list[list[str]]) -> TensorUpdateResult:
        new_sequences_cloned = _clone_sequences(new_sequences)
        delta = compute_sequence_delta(self.sequences, new_sequences_cloned)
        if delta.is_noop:
            return TensorUpdateResult(
                mode="noop",
                delta=delta,
                anchor_vocab_size=len(self.idx2token),
                oov_tokens=(),
                bigram_vector_size=self.bigram_vector.size,
                trigram_vector_size=self.trigram_vector.size,
            )

        oov_tokens = tuple(sorted(tok for tok in delta.added_tokens if tok not in self.token2idx))
        update_mode = "incremental"
        if delta.sequence_count_changed:
            rebuilt = type(self).from_sequences(
                new_sequences_cloned,
                config=self.config,
                ngram_basis=self.ngram_basis,
            )
            self._replace_with(rebuilt)
            return TensorUpdateResult(
                mode="full_rebuild",
                delta=delta,
                anchor_vocab_size=len(self.idx2token),
                oov_tokens=oov_tokens,
                bigram_vector_size=self.bigram_vector.size,
                trigram_vector_size=self.trigram_vector.size,
            )
        if oov_tokens:
            if len(self.idx2token) + len(oov_tokens) <= self.config.max_vocab:
                self._extend_anchor(oov_tokens)
                update_mode = "anchor_extend"
            else:
                rebuilt = type(self).from_sequences(
                    new_sequences_cloned,
                    config=self.config,
                    ngram_basis=self.ngram_basis,
                )
                self._replace_with(rebuilt)
                return TensorUpdateResult(
                    mode="full_rebuild",
                    delta=delta,
                    anchor_vocab_size=len(self.idx2token),
                    oov_tokens=oov_tokens,
                    bigram_vector_size=self.bigram_vector.size,
                    trigram_vector_size=self.trigram_vector.size,
                )

        for idx in delta.changed_indices:
            old_seq = self.sequences[idx]
            new_seq = new_sequences_cloned[idx]

            _sequence_cooccurrence_update(
                self.cooccurrence_counts,
                old_seq,
                self.token2idx,
                window=self.config.cooccurrence_window,
                sign=-1,
            )
            _sequence_cooccurrence_update(
                self.cooccurrence_counts,
                new_seq,
                self.token2idx,
                window=self.config.cooccurrence_window,
                sign=1,
            )

            _sequence_positional_update(
                self.positional_counts,
                self.positional_initial_counts,
                self.positional_final_counts,
                self.positional_pos_sum,
                self.positional_pos_sq_sum,
                old_seq,
                self.token2idx,
                sign=-1,
            )
            _sequence_positional_update(
                self.positional_counts,
                self.positional_initial_counts,
                self.positional_final_counts,
                self.positional_pos_sum,
                self.positional_pos_sq_sum,
                new_seq,
                self.token2idx,
                sign=1,
            )

            self.token_counts.subtract(Counter(old_seq))
            self.token_counts += Counter(new_seq)
            self.bigram_counts.subtract(_sequence_ngrams(old_seq, 2))
            self.bigram_counts += _sequence_ngrams(new_seq, 2)
            self.trigram_counts.subtract(_sequence_ngrams(old_seq, 3))
            self.trigram_counts += _sequence_ngrams(new_seq, 3)
            self.sequences[idx] = list(new_seq)

        self.bigram_counts = Counter({k: v for k, v in self.bigram_counts.items() if v > 0})
        self.trigram_counts = Counter({k: v for k, v in self.trigram_counts.items() if v > 0})
        self.token_counts = Counter({k: v for k, v in self.token_counts.items() if v > 0})
        self._refresh_views()

        return TensorUpdateResult(
            mode=update_mode,
            delta=delta,
            anchor_vocab_size=len(self.idx2token),
            oov_tokens=oov_tokens if update_mode == "anchor_extend" else (),
            bigram_vector_size=self.bigram_vector.size,
            trigram_vector_size=self.trigram_vector.size,
        )


__all__ = [
    "IncrementalFingerprintTensorState",
    "NgramVectorBasis",
    "SequenceDelta",
    "TensorStateConfig",
    "TensorUpdateResult",
    "compute_sequence_delta",
]
