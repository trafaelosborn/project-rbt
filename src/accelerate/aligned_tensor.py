"""
Alignment helpers for the Phase 2 Fortran acceleration layer.

Phase 2 keeps the same source-vocabulary alignment rule used by the reinforced
gradient path: current/source vocab is the anchor space, and reference tensors
are projected into that space before numeric comparison.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.accelerate.tensor_layout import FingerprintTensorLayout

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MATRICES_DIR = PROJECT_ROOT / "data" / "matrices"


def _load_meta(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _component_paths(label: str, matrices_dir: Path = MATRICES_DIR) -> dict[str, Path]:
    return {
        "cooccurrence_matrix": matrices_dir / f"{label}_cooccurrence.npy",
        "cooccurrence_meta": matrices_dir / f"{label}_cooccurrence_meta.json",
        "positional_matrix": matrices_dir / f"{label}_positional.npy",
        "positional_meta": matrices_dir / f"{label}_positional_meta.json",
    }


def align_square_matrix_to_anchor_tokens(
    *,
    anchor_tokens: list[str],
    matrix: np.ndarray,
    token2idx: dict[str, int],
    dtype: np.dtype = np.float64,
) -> np.ndarray:
    anchor_size = len(anchor_tokens)
    aligned = np.zeros((anchor_size, anchor_size), dtype=dtype, order="F")
    if anchor_size == 0:
        return aligned

    ref_indices = np.array([token2idx.get(tok, -1) for tok in anchor_tokens], dtype=np.int64)
    valid_mask = ref_indices >= 0
    if not valid_mask.any():
        return aligned

    anchor_valid = np.flatnonzero(valid_mask)
    ref_valid = ref_indices[valid_mask]
    aligned[np.ix_(anchor_valid, anchor_valid)] = np.asarray(matrix, dtype=dtype)[np.ix_(ref_valid, ref_valid)]
    return aligned


def align_feature_matrix_to_anchor_tokens(
    *,
    anchor_tokens: list[str],
    matrix: np.ndarray,
    token2idx: dict[str, int],
    dtype: np.dtype = np.float64,
) -> np.ndarray:
    matrix_arr = np.asarray(matrix, dtype=dtype)
    feature_width = matrix_arr.shape[1]
    aligned = np.zeros((len(anchor_tokens), feature_width), dtype=dtype, order="F")
    if not anchor_tokens:
        return aligned

    ref_indices = np.array([token2idx.get(tok, -1) for tok in anchor_tokens], dtype=np.int64)
    valid_mask = ref_indices >= 0
    if not valid_mask.any():
        return aligned

    anchor_valid = np.flatnonzero(valid_mask)
    ref_valid = ref_indices[valid_mask]
    aligned[anchor_valid] = matrix_arr[ref_valid]
    return aligned


@dataclass(frozen=True)
class AlignedTensorPair:
    anchor_label: str
    reference_label: str
    anchor_tokens: list[str]
    current_cooccurrence: np.ndarray
    reference_cooccurrence: np.ndarray
    current_positional: np.ndarray
    reference_positional: np.ndarray
    layout: FingerprintTensorLayout
    current_tensor: np.ndarray
    reference_tensor: np.ndarray
    current_paths: dict[str, str]
    reference_paths: dict[str, str]

    def manifest(self) -> dict:
        return {
            "anchor_label": self.anchor_label,
            "reference_label": self.reference_label,
            "anchor_vocab_size": len(self.anchor_tokens),
            "positional_width": int(self.current_positional.shape[1]),
            "layout": self.layout.manifest(),
            "current_paths": self.current_paths,
            "reference_paths": self.reference_paths,
        }


def load_aligned_tensor_pair(
    *,
    anchor_label: str,
    reference_label: str,
    matrices_dir: Path = MATRICES_DIR,
) -> AlignedTensorPair:
    current_paths = _component_paths(anchor_label, matrices_dir)
    reference_paths = _component_paths(reference_label, matrices_dir)

    current_cooc = np.load(current_paths["cooccurrence_matrix"])
    reference_cooc = np.load(reference_paths["cooccurrence_matrix"])
    current_pos = np.load(current_paths["positional_matrix"])
    reference_pos = np.load(reference_paths["positional_matrix"])

    current_cooc_meta = _load_meta(current_paths["cooccurrence_meta"])
    reference_cooc_meta = _load_meta(reference_paths["cooccurrence_meta"])
    current_pos_meta = _load_meta(current_paths["positional_meta"])
    reference_pos_meta = _load_meta(reference_paths["positional_meta"])

    anchor_tokens = list(current_cooc_meta["idx2token"])

    aligned_current_cooc = align_square_matrix_to_anchor_tokens(
        anchor_tokens=anchor_tokens,
        matrix=current_cooc,
        token2idx=current_cooc_meta["token2idx"],
    )
    aligned_reference_cooc = align_square_matrix_to_anchor_tokens(
        anchor_tokens=anchor_tokens,
        matrix=reference_cooc,
        token2idx=reference_cooc_meta["token2idx"],
    )
    aligned_current_pos = align_feature_matrix_to_anchor_tokens(
        anchor_tokens=anchor_tokens,
        matrix=current_pos,
        token2idx=current_pos_meta["token2idx"],
    )
    aligned_reference_pos = align_feature_matrix_to_anchor_tokens(
        anchor_tokens=anchor_tokens,
        matrix=reference_pos,
        token2idx=reference_pos_meta["token2idx"],
    )

    layout = FingerprintTensorLayout(
        vocab_size=len(anchor_tokens),
        positional_width=aligned_current_pos.shape[1],
        bigram_profile_size=0,
        trigram_profile_size=0,
    )
    current_tensor = layout.pack(
        cooccurrence=aligned_current_cooc,
        positional=aligned_current_pos,
        bigram_profile=np.zeros(0, dtype=np.float64),
        trigram_profile=np.zeros(0, dtype=np.float64),
    )
    reference_tensor = layout.pack(
        cooccurrence=aligned_reference_cooc,
        positional=aligned_reference_pos,
        bigram_profile=np.zeros(0, dtype=np.float64),
        trigram_profile=np.zeros(0, dtype=np.float64),
    )

    return AlignedTensorPair(
        anchor_label=anchor_label,
        reference_label=reference_label,
        anchor_tokens=anchor_tokens,
        current_cooccurrence=aligned_current_cooc,
        reference_cooccurrence=aligned_reference_cooc,
        current_positional=aligned_current_pos,
        reference_positional=aligned_reference_pos,
        layout=layout,
        current_tensor=current_tensor,
        reference_tensor=reference_tensor,
        current_paths={k: str(v) for k, v in current_paths.items()},
        reference_paths={k: str(v) for k, v in reference_paths.items()},
    )


__all__ = [
    "AlignedTensorPair",
    "align_feature_matrix_to_anchor_tokens",
    "align_square_matrix_to_anchor_tokens",
    "load_aligned_tensor_pair",
]
