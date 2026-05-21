"""
Fortran-guided batch landscape hints for the v4 reinforced engine.

This module keeps the methodology intact:
    - Python still owns mutation operators, scoring, coherence, and acceptance.
    - Fortran only computes a dense tensor-space adjustment batch.
    - Python then uses a Hungarian frontier over that batch to pick a diverse
      set of guidance hints for the next proposal cycle.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.accelerate.incremental_tensor_state import (
    IncrementalFingerprintTensorState,
    TensorStateConfig,
)
from src.accelerate.aligned_tensor import (
    align_feature_matrix_to_anchor_tokens,
    align_square_matrix_to_anchor_tokens,
)
from src.accelerate.fortran_batch import (
    COMPONENT_COOCCURRENCE,
    COMPONENT_LABELS,
    COMPONENT_POSITIONAL,
    AdjustmentCandidateBatch,
    fortran_top_adjustments,
    numpy_top_adjustments,
)
from src.fingerprint.cooccurrence import (
    DEFAULT_WINDOW,
    MAX_VOCAB,
    build_vocab,
    count_cooccurrences,
    l2_normalize_rows,
)
from src.fingerprint.positional import PositionalAccumulator

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MATRICES_DIR = PROJECT_ROOT / "data" / "matrices"


@dataclass(frozen=True)
class BatchGuidanceConfig:
    """Config for the optional v4 batch-guidance path."""

    backend: str = "auto"
    top_k: int = 512
    max_assignments: int = 24
    hotspot_token_limit: int = 48
    hotspot_pair_limit: int = 24
    max_vocab: int = MAX_VOCAB
    cooccurrence_window: int = DEFAULT_WINDOW
    module_name: str = "rbt_distance_kernels"
    build_dir: str | None = None
    force_rebuild: bool = False

    def to_dict(self) -> dict:
        return {
            "backend": self.backend,
            "top_k": self.top_k,
            "max_assignments": self.max_assignments,
            "hotspot_token_limit": self.hotspot_token_limit,
            "hotspot_pair_limit": self.hotspot_pair_limit,
            "max_vocab": self.max_vocab,
            "cooccurrence_window": self.cooccurrence_window,
            "module_name": self.module_name,
            "build_dir": self.build_dir,
            "force_rebuild": self.force_rebuild,
        }


@dataclass(frozen=True)
class ReferenceTensorSlices:
    label: str
    cooccurrence: np.ndarray
    positional: np.ndarray
    cooccurrence_token2idx: dict[str, int]
    positional_token2idx: dict[str, int]


@dataclass(frozen=True)
class GuidanceAdjustment:
    rank: int
    component_id: int
    component_name: str
    row_index: int
    col_index: int
    row_token: str
    col_token: str | None
    signed_delta: float
    abs_score: float

    def to_dict(self) -> dict:
        return {
            "rank": self.rank,
            "component_id": self.component_id,
            "component_name": self.component_name,
            "row_index": self.row_index,
            "col_index": self.col_index,
            "row_token": self.row_token,
            "col_token": self.col_token,
            "signed_delta": round(self.signed_delta, 6),
            "abs_score": round(self.abs_score, 6),
        }


@dataclass(frozen=True)
class BatchGuidance:
    backend_used: str
    anchor_vocab_size: int
    batch_size: int
    selected_adjustments: tuple[GuidanceAdjustment, ...]
    hotspot_token_weights: tuple[tuple[str, float], ...]
    hotspot_pairs: tuple[tuple[str, str, float, float], ...]
    positional_targets: dict[str, dict[int, float]]

    @property
    def selected_count(self) -> int:
        return len(self.selected_adjustments)

    @property
    def hotspot_tokens(self) -> tuple[str, ...]:
        return tuple(token for token, _ in self.hotspot_token_weights)

    def token_weight_map(self) -> dict[str, float]:
        return {token: weight for token, weight in self.hotspot_token_weights}

    def diagnostics(self, *, top_tokens: int = 8, top_pairs: int = 4, top_adjustments: int = 6) -> dict:
        return {
            "batch_guidance_backend": self.backend_used,
            "batch_guidance_anchor_vocab_size": self.anchor_vocab_size,
            "batch_guidance_batch_size": self.batch_size,
            "batch_guidance_selected_count": self.selected_count,
            "batch_guidance_hotspot_tokens": [
                {"token": token, "weight": round(weight, 6)}
                for token, weight in self.hotspot_token_weights[:top_tokens]
            ],
            "batch_guidance_hotspot_pairs": [
                {
                    "row_token": row_tok,
                    "col_token": col_tok,
                    "signed_delta": round(delta, 6),
                    "abs_score": round(score, 6),
                }
                for row_tok, col_tok, delta, score in self.hotspot_pairs[:top_pairs]
            ],
            "batch_guidance_selected_adjustments": [
                item.to_dict() for item in self.selected_adjustments[:top_adjustments]
            ],
        }


def _load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def load_reference_tensor_slices(
    reference_label: str = "latin",
    *,
    matrices_dir: Path = MATRICES_DIR,
) -> ReferenceTensorSlices:
    cooc_path = matrices_dir / f"{reference_label}_cooccurrence.npy"
    cooc_meta_path = matrices_dir / f"{reference_label}_cooccurrence_meta.json"
    pos_path = matrices_dir / f"{reference_label}_positional.npy"
    pos_meta_path = matrices_dir / f"{reference_label}_positional_meta.json"

    cooc_meta = _load_json(cooc_meta_path)
    pos_meta = _load_json(pos_meta_path)

    return ReferenceTensorSlices(
        label=reference_label,
        cooccurrence=np.load(cooc_path),
        positional=np.load(pos_path),
        cooccurrence_token2idx=cooc_meta["token2idx"],
        positional_token2idx=pos_meta["token2idx"],
    )


def build_current_tensor_slices(
    sequences: list[list[str]],
    *,
    max_vocab: int = MAX_VOCAB,
    cooccurrence_window: int = DEFAULT_WINDOW,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    token2idx, idx2token = build_vocab(sequences, max_vocab=max_vocab)

    counts = count_cooccurrences(sequences, token2idx, window=cooccurrence_window)
    current_cooccurrence = np.asfortranarray(l2_normalize_rows(counts), dtype=np.float64)

    positional_accumulator = PositionalAccumulator(token2idx)
    for seq in sequences:
        positional_accumulator.ingest_sequence(seq)
    current_positional = np.asfortranarray(positional_accumulator.to_matrix(), dtype=np.float64)

    return idx2token, current_cooccurrence, current_positional


def align_reference_to_anchor(
    *,
    anchor_tokens: list[str],
    reference: ReferenceTensorSlices,
) -> tuple[np.ndarray, np.ndarray]:
    aligned_cooccurrence = align_square_matrix_to_anchor_tokens(
        anchor_tokens=anchor_tokens,
        matrix=reference.cooccurrence,
        token2idx=reference.cooccurrence_token2idx,
    )
    aligned_positional = align_feature_matrix_to_anchor_tokens(
        anchor_tokens=anchor_tokens,
        matrix=reference.positional,
        token2idx=reference.positional_token2idx,
    )
    return aligned_cooccurrence, aligned_positional


def select_hungarian_frontier(
    batch: AdjustmentCandidateBatch,
    *,
    max_assignments: int,
) -> list[int]:
    if batch.size == 0 or max_assignments <= 0:
        return []

    best_idx_by_slot: dict[tuple[int, tuple[int, int]], int] = {}
    best_score_by_slot: dict[tuple[int, tuple[int, int]], float] = {}

    for idx in range(batch.size):
        row_key = int(batch.row_indices[idx])
        col_key = (int(batch.component_ids[idx]), int(batch.col_indices[idx]))
        slot_key = (row_key, col_key)
        score = float(batch.abs_scores[idx])
        if score > best_score_by_slot.get(slot_key, -1.0):
            best_score_by_slot[slot_key] = score
            best_idx_by_slot[slot_key] = idx

    row_keys = sorted({slot_key[0] for slot_key in best_idx_by_slot})
    col_keys = sorted({slot_key[1] for slot_key in best_idx_by_slot})
    if not row_keys or not col_keys:
        return []

    row_pos = {key: i for i, key in enumerate(row_keys)}
    col_pos = {key: i for i, key in enumerate(col_keys)}

    score_matrix = np.zeros((len(row_keys), len(col_keys)), dtype=np.float64)
    index_matrix = -np.ones((len(row_keys), len(col_keys)), dtype=np.int64)

    for slot_key, idx in best_idx_by_slot.items():
        row_key, col_key = slot_key
        i = row_pos[row_key]
        j = col_pos[col_key]
        score_matrix[i, j] = best_score_by_slot[slot_key]
        index_matrix[i, j] = idx

    max_score = float(score_matrix.max(initial=0.0))
    cost_matrix = max_score - score_matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    selected: list[tuple[float, int]] = []
    for i, j in zip(row_ind.tolist(), col_ind.tolist()):
        idx = int(index_matrix[i, j])
        if idx < 0:
            continue
        score = float(score_matrix[i, j])
        if score <= 0.0:
            continue
        selected.append((score, idx))

    selected.sort(key=lambda item: (-item[0], item[1]))
    return [idx for _, idx in selected[:max_assignments]]


def _build_guidance_from_indices(
    *,
    batch: AdjustmentCandidateBatch,
    selected_indices: list[int],
    anchor_tokens: list[str],
    backend_used: str,
    hotspot_token_limit: int,
    hotspot_pair_limit: int,
) -> BatchGuidance:
    token_weights: defaultdict[str, float] = defaultdict(float)
    positional_targets: defaultdict[str, dict[int, float]] = defaultdict(dict)
    hotspot_pairs: list[tuple[str, str, float, float]] = []
    selected_adjustments: list[GuidanceAdjustment] = []

    for rank, idx in enumerate(selected_indices, start=1):
        component_id = int(batch.component_ids[idx])
        component_name = COMPONENT_LABELS.get(component_id, "unknown")
        row_index = int(batch.row_indices[idx])
        col_index = int(batch.col_indices[idx])
        signed_delta = float(batch.signed_deltas[idx])
        abs_score = float(batch.abs_scores[idx])
        row_token = anchor_tokens[row_index]
        col_token = anchor_tokens[col_index] if component_id == COMPONENT_COOCCURRENCE else None

        selected_adjustments.append(
            GuidanceAdjustment(
                rank=rank,
                component_id=component_id,
                component_name=component_name,
                row_index=row_index,
                col_index=col_index,
                row_token=row_token,
                col_token=col_token,
                signed_delta=signed_delta,
                abs_score=abs_score,
            )
        )

        token_weights[row_token] += abs_score

        if component_id == COMPONENT_COOCCURRENCE and col_token is not None:
            token_weights[col_token] += 0.5 * abs_score
            hotspot_pairs.append((row_token, col_token, signed_delta, abs_score))
        elif component_id == COMPONENT_POSITIONAL:
            positional_targets[row_token][col_index] = signed_delta

    hotspot_token_weights = tuple(
        sorted(token_weights.items(), key=lambda item: (-item[1], item[0]))[:hotspot_token_limit]
    )
    hotspot_pairs_sorted = tuple(
        sorted(hotspot_pairs, key=lambda item: (-abs(item[3]), item[0], item[1]))[:hotspot_pair_limit]
    )

    return BatchGuidance(
        backend_used=backend_used,
        anchor_vocab_size=len(anchor_tokens),
        batch_size=batch.size,
        selected_adjustments=tuple(selected_adjustments),
        hotspot_token_weights=hotspot_token_weights,
        hotspot_pairs=hotspot_pairs_sorted,
        positional_targets=dict(positional_targets),
    )


class TensorBatchGuidanceBuilder:
    """Build v4 guidance hints from the current bridge tensor and Latin reference."""

    def __init__(
        self,
        config: BatchGuidanceConfig | None = None,
        *,
        reference_label: str = "latin",
        reference_slices: ReferenceTensorSlices | None = None,
    ) -> None:
        self.config = config or BatchGuidanceConfig()
        self.reference_label = reference_label
        self.reference_slices = reference_slices or load_reference_tensor_slices(reference_label)
        self._build_dir = Path(self.config.build_dir) if self.config.build_dir else None

    def tensor_state_config(self) -> TensorStateConfig:
        return TensorStateConfig(
            max_vocab=self.config.max_vocab,
            cooccurrence_window=self.config.cooccurrence_window,
        )

    def build_initial_state(
        self,
        sequences: list[list[str]],
    ) -> IncrementalFingerprintTensorState:
        return IncrementalFingerprintTensorState.from_sequences(
            sequences,
            config=self.tensor_state_config(),
            ngram_basis=None,
        )

    def _compute_batch(
        self,
        current_cooccurrence: np.ndarray,
        reference_cooccurrence: np.ndarray,
        current_positional: np.ndarray,
        reference_positional: np.ndarray,
    ) -> tuple[AdjustmentCandidateBatch, str]:
        backend = self.config.backend
        if backend == "numpy":
            return (
                numpy_top_adjustments(
                    current_cooccurrence,
                    reference_cooccurrence,
                    current_positional,
                    reference_positional,
                    top_k=self.config.top_k,
                ),
                "numpy",
            )
        if backend == "fortran":
            return (
                fortran_top_adjustments(
                    current_cooccurrence,
                    reference_cooccurrence,
                    current_positional,
                    reference_positional,
                    top_k=self.config.top_k,
                    build_dir=self._build_dir,
                    module_name=self.config.module_name,
                    force_rebuild=self.config.force_rebuild,
                ),
                "fortran",
            )

        try:
            batch = fortran_top_adjustments(
                current_cooccurrence,
                reference_cooccurrence,
                current_positional,
                reference_positional,
                top_k=self.config.top_k,
                build_dir=self._build_dir,
                module_name=self.config.module_name,
                force_rebuild=self.config.force_rebuild,
            )
            return batch, "fortran"
        except Exception as exc:  # pragma: no cover - exercised only when toolchain is missing/broken
            log.warning("Falling back to NumPy batch guidance after Fortran failure: %s", exc)
            return (
                numpy_top_adjustments(
                    current_cooccurrence,
                    reference_cooccurrence,
                    current_positional,
                    reference_positional,
                    top_k=self.config.top_k,
                ),
                "numpy",
            )

    def _build_from_components(
        self,
        *,
        anchor_tokens: list[str],
        current_cooccurrence: np.ndarray,
        current_positional: np.ndarray,
    ) -> BatchGuidance:
        reference_cooccurrence, reference_positional = align_reference_to_anchor(
            anchor_tokens=anchor_tokens,
            reference=self.reference_slices,
        )

        batch, backend_used = self._compute_batch(
            current_cooccurrence,
            reference_cooccurrence,
            current_positional,
            reference_positional,
        )
        selected_indices = select_hungarian_frontier(
            batch,
            max_assignments=self.config.max_assignments,
        )
        return _build_guidance_from_indices(
            batch=batch,
            selected_indices=selected_indices,
            anchor_tokens=anchor_tokens,
            backend_used=backend_used,
            hotspot_token_limit=self.config.hotspot_token_limit,
            hotspot_pair_limit=self.config.hotspot_pair_limit,
        )

    def build(self, sequences: list[list[str]]) -> BatchGuidance:
        anchor_tokens, current_cooccurrence, current_positional = build_current_tensor_slices(
            sequences,
            max_vocab=self.config.max_vocab,
            cooccurrence_window=self.config.cooccurrence_window,
        )
        return self._build_from_components(
            anchor_tokens=anchor_tokens,
            current_cooccurrence=current_cooccurrence,
            current_positional=current_positional,
        )

    def build_from_state(self, state: IncrementalFingerprintTensorState) -> BatchGuidance:
        return self._build_from_components(
            anchor_tokens=state.idx2token,
            current_cooccurrence=state.cooccurrence_matrix,
            current_positional=state.positional_matrix,
        )


__all__ = [
    "BatchGuidance",
    "BatchGuidanceConfig",
    "GuidanceAdjustment",
    "ReferenceTensorSlices",
    "TensorBatchGuidanceBuilder",
    "align_reference_to_anchor",
    "build_current_tensor_slices",
    "load_reference_tensor_slices",
    "select_hungarian_frontier",
]
