import math

import pytest

from src.validation import static_feature_matrix as sfm


def test_subsample_sequences_is_deterministic_and_hits_budget():
    sequences = [["a", "b", "c"], ["d", "e"], ["f", "g", "h", "i"]]

    first = sfm.subsample_sequences(
        sequences,
        token_budget=5,
        seed=123,
        corpus_id="toy",
    )
    second = sfm.subsample_sequences(
        sequences,
        token_budget=5,
        seed=123,
        corpus_id="toy",
    )

    assert first == second
    assert sum(len(seq) for seq in first) == 5


def test_compute_feature_vector_emits_registered_and_auxiliary_features():
    sequences = [
        ["arma", "virumque", "cano"],
        ["troiae", "qui", "primus"],
        ["ab", "oris", "arma"],
        ["cano", "arma", "troiae"],
    ]

    features = sfm.compute_feature_vector(sequences)

    registered = {feature.column for feature in sfm.REGISTERED_FEATURES}
    auxiliary = {feature.column for feature in sfm.AUXILIARY_FEATURES}
    assert registered.issubset(features)
    assert auxiliary.issubset(features)
    assert len(registered) == 21
    assert features["f18_dislegomena_to_hapax"] > 0.0
    assert math.isfinite(features["f21_sentence_length_skewness"])
    assert math.isfinite(features["aux_sentence_length_excess_kurtosis"])


def test_spearman_correlations_use_average_ranks():
    rows = [
        {"a": 1.0, "b": 3.0, "c": 1.0},
        {"a": 2.0, "b": 2.0, "c": 1.0},
        {"a": 3.0, "b": 1.0, "c": 2.0},
    ]

    correlations = sfm.spearman_correlations(rows, ["a", "b", "c"])

    assert correlations[("a", "a")] == pytest.approx(1.0)
    assert correlations[("a", "b")] == pytest.approx(-1.0)
    assert correlations[("b", "a")] == pytest.approx(-1.0)
    assert correlations[("a", "c")] > 0.0


def test_select_in_loop_tier_demotes_collinear_later_candidate():
    correlations = {}
    for left in sfm.PROPOSED_IN_LOOP_COLUMNS:
        for right in sfm.PROPOSED_IN_LOOP_COLUMNS:
            correlations[(left, right)] = 1.0 if left == right else 0.0
    correlations[("f09_word_length_mean", "f02_top100_bigram_coverage")] = 0.91
    correlations[("f02_top100_bigram_coverage", "f09_word_length_mean")] = 0.91

    tiers = sfm.select_in_loop_tier(correlations, threshold=0.85)

    assert "f02_top100_bigram_coverage" in tiers["retained_in_loop"]
    assert "f09_word_length_mean" not in tiers["retained_in_loop"]
    assert tiers["demoted_from_in_loop"][0]["feature"] == "f09_word_length_mean"
