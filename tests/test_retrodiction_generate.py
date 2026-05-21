"""Tests for src.retrodiction.generate"""

import numpy as np
import pytest
from src.retrodiction.generate import BigramModel


def _simple_sequences():
    return [
        ["a", "b", "c"],
        ["a", "b", "a"],
        ["c", "b", "a"],
        ["a", "c", "b"],
    ] * 10


class TestBigramModelBuild:
    def test_vocab_size(self):
        model = BigramModel.from_sequences(_simple_sequences())
        assert model.V == 3
        assert set(model.vocab) == {"a", "b", "c"}

    def test_transitions_row_sum_to_one(self):
        model = BigramModel.from_sequences(_simple_sequences())
        row_sums = model.transitions.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-6)

    def test_transitions_nonnegative(self):
        model = BigramModel.from_sequences(_simple_sequences())
        assert (model.transitions >= 0).all()

    def test_max_vocab_cap(self):
        seqs = [[str(i) for i in range(20)]] * 5
        model = BigramModel.from_sequences(seqs, max_vocab=10)
        assert model.V == 10

    def test_seq_lengths_recorded(self):
        model = BigramModel.from_sequences(_simple_sequences())
        assert len(model.seq_lengths) > 0
        assert all(l > 0 for l in model.seq_lengths)

    def test_empty_sequences_handled(self):
        seqs = _simple_sequences() + [[]]
        model = BigramModel.from_sequences(seqs)
        assert model.V == 3


class TestBigramModelMix:
    def test_mix_rows_still_sum_to_one(self):
        model = BigramModel.from_sequences(_simple_sequences())
        mixed = model.mix_toward_uniform(0.1)
        row_sums = mixed.transitions.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-6)

    def test_mix_increases_entropy(self):
        model = BigramModel.from_sequences(_simple_sequences())
        mixed = model.mix_toward_uniform(0.5)
        assert mixed.bigram_entropy() > model.bigram_entropy()

    def test_full_mix_is_uniform(self):
        model = BigramModel.from_sequences(_simple_sequences())
        mixed = model.mix_toward_uniform(1.0)
        expected = np.full((model.V, model.V), 1.0 / model.V)
        assert np.allclose(mixed.transitions, expected, atol=1e-10)

    def test_zero_mix_unchanged(self):
        model = BigramModel.from_sequences(_simple_sequences())
        mixed = model.mix_toward_uniform(0.0)
        assert np.allclose(mixed.transitions, model.transitions)

    def test_vocab_preserved_after_mix(self):
        model = BigramModel.from_sequences(_simple_sequences())
        mixed = model.mix_toward_uniform(0.3)
        assert mixed.vocab == model.vocab
        assert mixed.token2idx == model.token2idx


class TestBigramModelSample:
    def test_sample_sequence_length(self):
        model = BigramModel.from_sequences(_simple_sequences())
        rng = np.random.default_rng(0)
        seq = model.sample_sequence(7, rng)
        assert len(seq) == 7

    def test_sample_sequence_tokens_in_vocab(self):
        model = BigramModel.from_sequences(_simple_sequences())
        rng = np.random.default_rng(0)
        for _ in range(20):
            seq = model.sample_sequence(10, rng)
            assert all(t in model.token2idx for t in seq)

    def test_sample_corpus_count(self):
        model = BigramModel.from_sequences(_simple_sequences())
        rng = np.random.default_rng(0)
        corpus = model.sample_corpus(50, rng)
        assert len(corpus) == 50

    def test_sample_corpus_reproducible(self):
        model = BigramModel.from_sequences(_simple_sequences())
        corpus1 = model.sample_corpus(20, np.random.default_rng(42))
        corpus2 = model.sample_corpus(20, np.random.default_rng(42))
        assert corpus1 == corpus2

    def test_sample_corpus_different_seeds(self):
        model = BigramModel.from_sequences(_simple_sequences())
        corpus1 = model.sample_corpus(20, np.random.default_rng(1))
        corpus2 = model.sample_corpus(20, np.random.default_rng(2))
        assert corpus1 != corpus2


class TestBigramEntropy:
    def test_uniform_model_max_entropy(self):
        model = BigramModel.from_sequences(_simple_sequences())
        uniform = model.mix_toward_uniform(1.0)
        import math
        expected = math.log(model.V)
        assert abs(uniform.bigram_entropy() - expected) < 1e-6

    def test_entropy_nonnegative(self):
        model = BigramModel.from_sequences(_simple_sequences())
        assert model.bigram_entropy() >= 0.0
