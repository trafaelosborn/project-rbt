"""
Phase 2 batch candidate generation for the Fortran acceleration layer.

Fortran computes the top adjustment landscape over aligned co-occurrence and
positional slices. Python keeps selection, Hungarian logic, and acceptance.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.accelerate.aligned_tensor import AlignedTensorPair, load_aligned_tensor_pair
from src.accelerate.fortran_distance import build_extension, detect_fortran_compiler, _load_extension_from_path

COMPONENT_COOCCURRENCE = 1
COMPONENT_POSITIONAL = 2
COMPONENT_LABELS = {
    COMPONENT_COOCCURRENCE: "cooccurrence",
    COMPONENT_POSITIONAL: "positional",
}


@dataclass(frozen=True)
class AdjustmentCandidateBatch:
    component_ids: np.ndarray
    row_indices: np.ndarray
    col_indices: np.ndarray
    signed_deltas: np.ndarray
    abs_scores: np.ndarray

    @property
    def size(self) -> int:
        return int(self.abs_scores.size)

    def to_records(self, limit: int | None = None) -> list[dict]:
        size = self.size if limit is None else min(self.size, limit)
        records = []
        for i in range(size):
            component_id = int(self.component_ids[i])
            records.append(
                {
                    "rank": i + 1,
                    "component_id": component_id,
                    "component_name": COMPONENT_LABELS.get(component_id, "unknown"),
                    "row_index": int(self.row_indices[i]),
                    "col_index": int(self.col_indices[i]),
                    "signed_delta": float(self.signed_deltas[i]),
                    "abs_score": float(self.abs_scores[i]),
                }
            )
        return records


@dataclass(frozen=True)
class BatchBenchmarkResult:
    top_k: int
    current_shape: tuple[int, int]
    positional_shape: tuple[int, int]
    python_seconds: float
    fortran_seconds: float | None
    speedup_vs_python: float | None
    compiler: str | None
    status: str
    notes: str | None = None

    def to_dict(self) -> dict:
        return {
            "top_k": self.top_k,
            "current_shape": list(self.current_shape),
            "positional_shape": list(self.positional_shape),
            "python_seconds": round(self.python_seconds, 6),
            "fortran_seconds": None if self.fortran_seconds is None else round(self.fortran_seconds, 6),
            "speedup_vs_python": None
            if self.speedup_vs_python is None
            else round(self.speedup_vs_python, 6),
            "compiler": self.compiler,
            "status": self.status,
            "notes": self.notes,
        }


def _normalize_square_matrix(matrix: np.ndarray, label: str) -> np.ndarray:
    arr = np.asarray(matrix)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"{label} must be a square 2D matrix, got shape {arr.shape!r}")
    return np.asfortranarray(arr, dtype=np.float64)


def _normalize_feature_matrix(matrix: np.ndarray, label: str) -> np.ndarray:
    arr = np.asarray(matrix)
    if arr.ndim != 2:
        raise ValueError(f"{label} must be a 2D matrix, got shape {arr.shape!r}")
    return np.asfortranarray(arr, dtype=np.float64)


def _validate_batch_inputs(
    cooc_current: np.ndarray,
    cooc_reference: np.ndarray,
    pos_current: np.ndarray,
    pos_reference: np.ndarray,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if top_k <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}")

    cooc_current_arr = _normalize_square_matrix(cooc_current, "cooc_current")
    cooc_reference_arr = _normalize_square_matrix(cooc_reference, "cooc_reference")
    pos_current_arr = _normalize_feature_matrix(pos_current, "pos_current")
    pos_reference_arr = _normalize_feature_matrix(pos_reference, "pos_reference")

    if cooc_current_arr.shape != cooc_reference_arr.shape:
        raise ValueError(
            f"cooccurrence shapes must match: {cooc_current_arr.shape!r} != {cooc_reference_arr.shape!r}"
        )
    if pos_current_arr.shape != pos_reference_arr.shape:
        raise ValueError(
            f"positional shapes must match: {pos_current_arr.shape!r} != {pos_reference_arr.shape!r}"
        )
    if pos_current_arr.shape[0] != cooc_current_arr.shape[0]:
        raise ValueError(
            "cooccurrence and positional matrices must share the same row vocabulary size: "
            f"{cooc_current_arr.shape[0]!r} != {pos_current_arr.shape[0]!r}"
        )
    return cooc_current_arr, cooc_reference_arr, pos_current_arr, pos_reference_arr


def _top_component_candidates(
    delta: np.ndarray,
    *,
    component_id: int,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    flat_delta = np.ravel(np.asarray(delta, dtype=np.float64), order="F")
    flat_abs = np.abs(flat_delta)
    candidate_count = min(top_k, flat_abs.size)
    if candidate_count == 0:
        empty = np.zeros(0, dtype=np.int64)
        empty_f = np.zeros(0, dtype=np.float64)
        return empty, empty, empty, empty_f, empty_f

    top_idx = np.argpartition(flat_abs, -candidate_count)[-candidate_count:]
    rows, cols = np.unravel_index(top_idx, delta.shape, order="F")

    component_ids = np.full(candidate_count, component_id, dtype=np.int64)
    row_indices = rows.astype(np.int64)
    col_indices = cols.astype(np.int64)
    signed_deltas = flat_delta[top_idx].astype(np.float64)
    abs_scores = flat_abs[top_idx].astype(np.float64)

    return component_ids, row_indices, col_indices, signed_deltas, abs_scores


def _sort_candidate_arrays(
    component_ids: np.ndarray,
    row_indices: np.ndarray,
    col_indices: np.ndarray,
    signed_deltas: np.ndarray,
    abs_scores: np.ndarray,
    *,
    top_k: int,
) -> AdjustmentCandidateBatch:
    if abs_scores.size == 0:
        empty_i = np.zeros(0, dtype=np.int64)
        empty_f = np.zeros(0, dtype=np.float64)
        return AdjustmentCandidateBatch(empty_i, empty_i, empty_i, empty_f, empty_f)

    order = np.lexsort((component_ids, col_indices, row_indices, -abs_scores))
    order = order[: min(top_k, order.size)]
    return AdjustmentCandidateBatch(
        component_ids=component_ids[order],
        row_indices=row_indices[order],
        col_indices=col_indices[order],
        signed_deltas=signed_deltas[order],
        abs_scores=abs_scores[order],
    )


def numpy_top_adjustments(
    cooc_current: np.ndarray,
    cooc_reference: np.ndarray,
    pos_current: np.ndarray,
    pos_reference: np.ndarray,
    *,
    top_k: int,
) -> AdjustmentCandidateBatch:
    cooc_current_arr, cooc_reference_arr, pos_current_arr, pos_reference_arr = _validate_batch_inputs(
        cooc_current, cooc_reference, pos_current, pos_reference, top_k
    )

    cooc_delta = cooc_reference_arr - cooc_current_arr
    pos_delta = pos_reference_arr - pos_current_arr

    cooc_candidates = _top_component_candidates(cooc_delta, component_id=COMPONENT_COOCCURRENCE, top_k=top_k)
    pos_candidates = _top_component_candidates(pos_delta, component_id=COMPONENT_POSITIONAL, top_k=top_k)

    component_ids = np.concatenate((cooc_candidates[0], pos_candidates[0]))
    row_indices = np.concatenate((cooc_candidates[1], pos_candidates[1]))
    col_indices = np.concatenate((cooc_candidates[2], pos_candidates[2]))
    signed_deltas = np.concatenate((cooc_candidates[3], pos_candidates[3]))
    abs_scores = np.concatenate((cooc_candidates[4], pos_candidates[4]))

    return _sort_candidate_arrays(
        component_ids,
        row_indices,
        col_indices,
        signed_deltas,
        abs_scores,
        top_k=top_k,
    )


def fortran_top_adjustments(
    cooc_current: np.ndarray,
    cooc_reference: np.ndarray,
    pos_current: np.ndarray,
    pos_reference: np.ndarray,
    *,
    top_k: int,
    build_dir: Path | None = None,
    module_name: str = "rbt_distance_kernels",
    force_rebuild: bool = False,
) -> AdjustmentCandidateBatch:
    cooc_current_arr, cooc_reference_arr, pos_current_arr, pos_reference_arr = _validate_batch_inputs(
        cooc_current, cooc_reference, pos_current, pos_reference, top_k
    )

    extension_path = build_extension(force=force_rebuild, build_dir=build_dir, module_name=module_name)
    module = _load_extension_from_path(extension_path, module_name=module_name)
    component_ids, row_indices, col_indices, signed_deltas, abs_scores, actual_k = module.top_adjustments_batch(
        cooc_current_arr,
        cooc_reference_arr,
        pos_current_arr,
        pos_reference_arr,
        int(top_k),
    )
    actual_k = int(actual_k)
    return AdjustmentCandidateBatch(
        component_ids=np.asarray(component_ids, dtype=np.int64)[:actual_k],
        row_indices=np.asarray(row_indices, dtype=np.int64)[:actual_k] - 1,
        col_indices=np.asarray(col_indices, dtype=np.int64)[:actual_k] - 1,
        signed_deltas=np.asarray(signed_deltas, dtype=np.float64)[:actual_k],
        abs_scores=np.asarray(abs_scores, dtype=np.float64)[:actual_k],
    )


def load_default_phase2_inputs(
    *,
    anchor_label: str = "french",
    reference_label: str = "latin",
) -> AlignedTensorPair:
    return load_aligned_tensor_pair(anchor_label=anchor_label, reference_label=reference_label)


def benchmark_top_adjustments(
    cooc_current: np.ndarray,
    cooc_reference: np.ndarray,
    pos_current: np.ndarray,
    pos_reference: np.ndarray,
    *,
    top_k: int = 512,
    repeats: int = 3,
    build_dir: Path | None = None,
    force_rebuild: bool = False,
) -> BatchBenchmarkResult:
    python_times: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        numpy_top_adjustments(cooc_current, cooc_reference, pos_current, pos_reference, top_k=top_k)
        python_times.append(time.perf_counter() - started)
    python_seconds = min(python_times)

    compiler = detect_fortran_compiler()
    if compiler is None:
        return BatchBenchmarkResult(
            top_k=top_k,
            current_shape=np.asarray(cooc_current).shape,
            positional_shape=np.asarray(pos_current).shape,
            python_seconds=python_seconds,
            fortran_seconds=None,
            speedup_vs_python=None,
            compiler=None,
            status="compiler_unavailable",
            notes="Python baseline measured; install a local Fortran compiler to benchmark batch top-k extraction.",
        )

    build_extension(force=force_rebuild, build_dir=build_dir)
    fortran_times: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        fortran_top_adjustments(
            cooc_current,
            cooc_reference,
            pos_current,
            pos_reference,
            top_k=top_k,
            build_dir=build_dir,
            force_rebuild=False,
        )
        fortran_times.append(time.perf_counter() - started)
    fortran_seconds = min(fortran_times)
    return BatchBenchmarkResult(
        top_k=top_k,
        current_shape=np.asarray(cooc_current).shape,
        positional_shape=np.asarray(pos_current).shape,
        python_seconds=python_seconds,
        fortran_seconds=fortran_seconds,
        speedup_vs_python=python_seconds / fortran_seconds if fortran_seconds else None,
        compiler=compiler,
        status="ok",
        notes=None,
    )


def benchmark_to_json(
    output_path: Path,
    *,
    anchor_label: str = "french",
    reference_label: str = "latin",
    top_k: int = 512,
    repeats: int = 3,
    build_dir: Path | None = None,
    force_rebuild: bool = False,
) -> dict:
    pair = load_default_phase2_inputs(anchor_label=anchor_label, reference_label=reference_label)
    result = benchmark_top_adjustments(
        pair.current_cooccurrence,
        pair.reference_cooccurrence,
        pair.current_positional,
        pair.reference_positional,
        top_k=top_k,
        repeats=repeats,
        build_dir=build_dir,
        force_rebuild=force_rebuild,
    )
    payload = {
        "anchor_label": anchor_label,
        "reference_label": reference_label,
        "pair_manifest": pair.manifest(),
        **result.to_dict(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return payload


__all__ = [
    "AdjustmentCandidateBatch",
    "BatchBenchmarkResult",
    "COMPONENT_COOCCURRENCE",
    "COMPONENT_LABELS",
    "COMPONENT_POSITIONAL",
    "benchmark_to_json",
    "benchmark_top_adjustments",
    "fortran_top_adjustments",
    "load_default_phase2_inputs",
    "numpy_top_adjustments",
]
