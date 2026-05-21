"""Tests for the optional v4 batch-guidance layer."""

from __future__ import annotations

import numpy as np

from src.accelerate.incremental_tensor_state import IncrementalFingerprintTensorState, TensorStateConfig
from src.accelerate.fortran_batch import AdjustmentCandidateBatch
from src.accelerate.v4_batch_guidance import (
    BatchGuidanceConfig,
    ReferenceTensorSlices,
    TensorBatchGuidanceBuilder,
    select_hungarian_frontier,
)


def test_select_hungarian_frontier_diversifies_rows_and_columns():
    batch = AdjustmentCandidateBatch(
        component_ids=np.array([1, 1, 1, 2], dtype=np.int64),
        row_indices=np.array([0, 0, 1, 1], dtype=np.int64),
        col_indices=np.array([0, 1, 1, 0], dtype=np.int64),
        signed_deltas=np.array([5.0, 4.5, 4.0, 3.0], dtype=np.float64),
        abs_scores=np.array([5.0, 4.5, 4.0, 3.0], dtype=np.float64),
    )

    selected = select_hungarian_frontier(batch, max_assignments=4)

    assert selected == [0, 2]


def test_tensor_batch_guidance_builder_numpy_builds_hotspots():
    reference = ReferenceTensorSlices(
        label="latin",
        cooccurrence=np.array(
            [
                [0.0, 1.0, 0.5],
                [1.0, 0.0, 0.2],
                [0.5, 0.2, 0.0],
            ],
            dtype=np.float64,
        ),
        positional=np.array(
            [
                [0.8, 0.0, 0.2, 0.1, 0.0, 1.0],
                [0.2, 0.2, 0.6, 0.5, 0.1, 0.8],
                [0.0, 0.7, 0.3, 0.9, 0.2, 0.6],
            ],
            dtype=np.float64,
        ),
        cooccurrence_token2idx={"de": 0, "forum": 1, "bellum": 2},
        positional_token2idx={"de": 0, "forum": 1, "bellum": 2},
    )
    builder = TensorBatchGuidanceBuilder(
        BatchGuidanceConfig(
            backend="numpy",
            top_k=6,
            max_assignments=3,
            hotspot_token_limit=4,
            hotspot_pair_limit=4,
            max_vocab=3,
        ),
        reference_slices=reference,
    )

    sequences = [
        ["de", "de", "forum"],
        ["de", "forum"],
        ["bellum", "de"],
        ["forum", "de"],
    ]
    guidance = builder.build(sequences)

    assert guidance.backend_used == "numpy"
    assert guidance.selected_count >= 1
    assert guidance.batch_size >= guidance.selected_count
    assert len(guidance.hotspot_tokens) >= 1
    assert set(guidance.hotspot_tokens).issubset({"de", "forum", "bellum"})
    assert guidance.diagnostics()["batch_guidance_selected_count"] == guidance.selected_count


def test_tensor_batch_guidance_builder_from_state_matches_sequence_build():
    reference = ReferenceTensorSlices(
        label="latin",
        cooccurrence=np.array(
            [
                [0.0, 1.0, 0.5],
                [1.0, 0.0, 0.2],
                [0.5, 0.2, 0.0],
            ],
            dtype=np.float64,
        ),
        positional=np.array(
            [
                [0.8, 0.0, 0.2, 0.1, 0.0, 1.0],
                [0.2, 0.2, 0.6, 0.5, 0.1, 0.8],
                [0.0, 0.7, 0.3, 0.9, 0.2, 0.6],
            ],
            dtype=np.float64,
        ),
        cooccurrence_token2idx={"de": 0, "forum": 1, "bellum": 2},
        positional_token2idx={"de": 0, "forum": 1, "bellum": 2},
    )
    builder = TensorBatchGuidanceBuilder(
        BatchGuidanceConfig(
            backend="numpy",
            top_k=6,
            max_assignments=3,
            hotspot_token_limit=4,
            hotspot_pair_limit=4,
            max_vocab=3,
            cooccurrence_window=2,
        ),
        reference_slices=reference,
    )
    sequences = [
        ["de", "de", "forum"],
        ["de", "forum"],
        ["bellum", "de"],
        ["forum", "de"],
    ]
    state = IncrementalFingerprintTensorState.from_sequences(
        sequences,
        config=TensorStateConfig(max_vocab=3, cooccurrence_window=2),
        ngram_basis=None,
    )

    direct = builder.build(sequences)
    from_state = builder.build_from_state(state)

    assert direct.backend_used == from_state.backend_used
    assert direct.anchor_vocab_size == from_state.anchor_vocab_size
    assert direct.hotspot_token_weights == from_state.hotspot_token_weights
    assert direct.hotspot_pairs == from_state.hotspot_pairs
    assert [item.to_dict() for item in direct.selected_adjustments] == [
        item.to_dict() for item in from_state.selected_adjustments
    ]
