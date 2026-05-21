"""Tests for the Phase 4 incremental tensor state scaffold."""

from __future__ import annotations

import numpy as np

from src.accelerate.incremental_tensor_state import (
    IncrementalFingerprintTensorState,
    NgramVectorBasis,
    TensorStateConfig,
    compute_sequence_delta,
)


def _base_sequences():
    return [
        ["de", "forum", "bellum"],
        ["de", "forum", "romanum"],
        ["bellum", "de", "forum"],
        ["romanum", "forum", "de"],
    ]


def _basis():
    return NgramVectorBasis(
        bigram_keys=(
            "de | forum",
            "forum | bellum",
            "forum | romanum",
            "bellum | de",
        ),
        trigram_keys=(
            "de | forum | bellum",
            "de | forum | romanum",
            "romanum | forum | de",
        ),
    )


def test_compute_sequence_delta_tracks_changed_sequences_and_tokens():
    before = _base_sequences()
    after = [
        ["de", "forum", "bellum"],
        ["de", "forumum", "romanum"],
        ["bellum", "de", "forum"],
        ["romanum", "forum", "de"],
    ]

    delta = compute_sequence_delta(before, after)

    assert delta.changed_indices == (1,)
    assert "forumum" in delta.added_tokens
    assert "forum" in delta.removed_tokens
    assert "de | forum | romanum" in delta.touched_trigrams


def test_incremental_update_matches_fresh_rebuild():
    config = TensorStateConfig(max_vocab=32, ngram_top_n=16)
    state = IncrementalFingerprintTensorState.from_sequences(
        _base_sequences(),
        config=config,
        ngram_basis=_basis(),
    )
    updated_sequences = [
        ["de", "forum", "bellum"],
        ["de", "forum", "romanum"],
        ["bellum", "forum", "de"],
        ["romanum", "forum", "de"],
    ]

    result = state.apply_sequences(updated_sequences)
    rebuilt = IncrementalFingerprintTensorState.from_sequences_with_anchor(
        updated_sequences,
        anchor_tokens=state.idx2token,
        config=config,
        ngram_basis=_basis(),
    )

    assert result.mode == "incremental"
    assert np.array_equal(state.cooccurrence_counts, rebuilt.cooccurrence_counts)
    assert np.allclose(state.cooccurrence_matrix, rebuilt.cooccurrence_matrix)
    assert np.allclose(state.positional_matrix, rebuilt.positional_matrix)
    assert np.allclose(state.bigram_vector, rebuilt.bigram_vector)
    assert np.allclose(state.trigram_vector, rebuilt.trigram_vector)
    assert np.allclose(state.tensor, rebuilt.tensor)


def test_apply_sequences_extends_anchor_when_oov_token_is_added_with_headroom():
    config = TensorStateConfig(max_vocab=32, ngram_top_n=16)
    state = IncrementalFingerprintTensorState.from_sequences(
        _base_sequences(),
        config=config,
        ngram_basis=_basis(),
    )
    updated_sequences = [
        ["de", "forum", "bellum"],
        ["de", "forum", "romanum"],
        ["bellum", "de", "novumtoken"],
        ["romanum", "forum", "de"],
    ]

    result = state.apply_sequences(updated_sequences)
    rebuilt = IncrementalFingerprintTensorState.from_sequences_with_anchor(
        updated_sequences,
        anchor_tokens=state.idx2token,
        config=config,
        ngram_basis=_basis(),
    )

    assert result.mode == "anchor_extend"
    assert "novumtoken" in result.oov_tokens
    assert "novumtoken" in state.token2idx
    assert state.sequences == updated_sequences
    assert np.array_equal(state.cooccurrence_counts, rebuilt.cooccurrence_counts)
    assert np.allclose(state.positional_matrix, rebuilt.positional_matrix)
    assert np.allclose(state.tensor, rebuilt.tensor)


def test_apply_sequences_falls_back_to_full_rebuild_when_anchor_is_full():
    config = TensorStateConfig(max_vocab=4, ngram_top_n=16)
    state = IncrementalFingerprintTensorState.from_sequences(
        _base_sequences(),
        config=config,
        ngram_basis=_basis(),
    )
    updated_sequences = [
        ["de", "forum", "bellum"],
        ["de", "forum", "romanum"],
        ["bellum", "de", "novumtoken"],
        ["romanum", "forum", "de"],
    ]

    result = state.apply_sequences(updated_sequences)

    assert result.mode == "full_rebuild"
    assert "novumtoken" in result.oov_tokens
    assert state.sequences == updated_sequences


def test_tensor_manifest_reports_ngram_vector_sizes():
    state = IncrementalFingerprintTensorState.from_sequences(
        _base_sequences(),
        config=TensorStateConfig(max_vocab=32, ngram_top_n=16),
        ngram_basis=_basis(),
    )

    manifest = state.manifest()

    assert manifest["anchor_vocab_size"] == len(state.idx2token)
    assert manifest["ngram_basis"]["bigram_size"] == len(_basis().bigram_keys)
    assert manifest["layout"]["components"][2]["name"] == "bigram_profile"


def test_incremental_update_keeps_tied_ngram_profiles_deterministic():
    sequences = [
        ["a", "b", "c", "d"],
        ["a", "c", "b", "d"],
        ["b", "a", "d", "c"],
        ["b", "c", "a", "d"],
    ]
    basis = NgramVectorBasis(
        bigram_keys=(
            "a | b",
            "a | c",
            "b | a",
            "b | c",
            "c | a",
            "c | b",
        ),
        trigram_keys=(
            "a | b | c",
            "a | c | b",
            "b | a | d",
            "b | c | a",
        ),
    )
    state = IncrementalFingerprintTensorState.from_sequences(
        sequences,
        config=TensorStateConfig(max_vocab=32, ngram_top_n=4),
        ngram_basis=basis,
    )
    updated_sequences = [list(seq) for seq in sequences]
    updated_sequences[0] = ["a", "c", "b", "d"]
    updated_sequences[1] = ["a", "b", "c", "d"]

    result = state.apply_sequences(updated_sequences)
    rebuilt = IncrementalFingerprintTensorState.from_sequences_with_anchor(
        updated_sequences,
        anchor_tokens=state.idx2token,
        config=TensorStateConfig(max_vocab=32, ngram_top_n=4),
        ngram_basis=basis,
    )

    assert result.mode == "incremental"
    assert np.allclose(state.bigram_vector, rebuilt.bigram_vector)
    assert np.allclose(state.trigram_vector, rebuilt.trigram_vector)
    assert np.allclose(state.tensor, rebuilt.tensor)
