"""
Unit tests for src.fingerprint.ngram
"""

import pytest
from collections import Counter
from src.fingerprint.ngram import (
    extract_ngrams,
    build_profile,
    type_token_ratio,
    validate_profile,
    NGRAM_SEP,
)


class TestExtractNgrams:
    def test_bigrams_basic(self):
        seqs = [["a", "b", "c"]]
        counts = extract_ngrams(seqs, 2)
        assert counts[("a", "b")] == 1
        assert counts[("b", "c")] == 1
        assert len(counts) == 2

    def test_trigrams_basic(self):
        seqs = [["a", "b", "c", "d"]]
        counts = extract_ngrams(seqs, 3)
        assert counts[("a", "b", "c")] == 1
        assert counts[("b", "c", "d")] == 1
        assert len(counts) == 2

    def test_no_ngram_spans_boundary(self):
        seq1 = ["a", "b"]
        seq2 = ["c", "d"]
        counts = extract_ngrams([seq1, seq2], 2)
        assert ("b", "c") not in counts

    def test_repeated_ngram_counted(self):
        seqs = [["a", "b", "a", "b"]]
        counts = extract_ngrams(seqs, 2)
        assert counts[("a", "b")] == 2
        assert counts[("b", "a")] == 1

    def test_empty_sequences(self):
        counts = extract_ngrams([[], []], 2)
        assert len(counts) == 0

    def test_sequence_shorter_than_n(self):
        seqs = [["a"]]
        counts = extract_ngrams(seqs, 2)
        assert len(counts) == 0

    def test_unigrams(self):
        seqs = [["a", "b", "a"]]
        counts = extract_ngrams(seqs, 1)
        assert counts[("a",)] == 2
        assert counts[("b",)] == 1

    def test_multiple_sequences_merged(self):
        seqs = [["a", "b"], ["a", "b"]]
        counts = extract_ngrams(seqs, 2)
        assert counts[("a", "b")] == 2


class TestBuildProfile:
    def test_frequencies_sum_to_one(self):
        counter = Counter({("a", "b"): 3, ("b", "c"): 1, ("c", "d"): 1})
        profile = build_profile(counter, top_n=10)
        total = sum(profile.values())
        assert abs(total - 1.0) < 1e-6

    def test_top_n_limits_entries(self):
        counter = Counter({(str(i), str(i+1)): i+1 for i in range(10)})
        profile = build_profile(counter, top_n=3)
        assert len(profile) == 3

    def test_most_frequent_has_highest_value(self):
        counter = Counter({("a", "b"): 100, ("c", "d"): 1})
        profile = build_profile(counter, top_n=2)
        key_ab = NGRAM_SEP.join(("a", "b"))
        key_cd = NGRAM_SEP.join(("c", "d"))
        assert profile[key_ab] > profile[key_cd]

    def test_empty_counter_returns_empty(self):
        profile = build_profile(Counter(), top_n=10)
        assert profile == {}

    def test_ngram_sep_used_as_key_separator(self):
        counter = Counter({("hello", "world"): 5})
        profile = build_profile(counter, top_n=10)
        expected_key = f"hello{NGRAM_SEP}world"
        assert expected_key in profile

    def test_top_n_larger_than_vocab(self):
        counter = Counter({("a", "b"): 2, ("c", "d"): 1})
        profile = build_profile(counter, top_n=100)
        assert len(profile) == 2


class TestTypeTokenRatio:
    def test_all_unique(self):
        seqs = [["a", "b", "c"]]
        assert type_token_ratio(seqs) == 1.0

    def test_all_same(self):
        seqs = [["a", "a", "a"]]
        assert abs(type_token_ratio(seqs) - 1/3) < 1e-9

    def test_empty(self):
        assert type_token_ratio([]) == 0.0

    def test_empty_sequences(self):
        assert type_token_ratio([[], []]) == 0.0

    def test_multiple_sequences(self):
        seqs = [["a", "b"], ["a", "c"]]
        # 3 types, 4 tokens → TTR = 0.75
        assert abs(type_token_ratio(seqs) - 0.75) < 1e-9

    def test_ttr_between_zero_and_one(self):
        seqs = [["the", "cat", "sat", "on", "the", "mat"]]
        ttr = type_token_ratio(seqs)
        assert 0.0 < ttr <= 1.0


class TestValidateProfile:
    def test_valid_profile_passes(self):
        profile = {"a | b": 0.6, "c | d": 0.4}
        validate_profile(profile, "bigram")  # should not raise

    def test_sum_not_one_raises(self):
        profile = {"a | b": 0.6, "c | d": 0.6}  # sums to 1.2
        with pytest.raises(ValueError):
            validate_profile(profile, "bigram")

    def test_empty_profile_does_not_raise(self):
        # Empty profile logs a warning but does not raise
        validate_profile({}, "bigram")
