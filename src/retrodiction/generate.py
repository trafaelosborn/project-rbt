"""
Bigram Language Model and Corpus Generator
==========================================
Purpose:
    Build a bigram language model from a token corpus, apply the backward
    transformation (mixing toward uniform distribution), and sample synthetic
    token sequences from the resulting model.

    This is the generative core of the retrodiction engine. Each iteration:
        1. The current model generates a synthetic corpus.
        2. The corpus is re-fingerprinted.
        3. The model is mixed slightly toward uniform (the backward step).
        4. Repeat.

    Mixing toward uniform increases bigram entropy — tokens become less
    predictable given their predecessor. This corresponds to moving toward
    freer word order (more synthetic grammar). The generated sequences at
    each stage are the intermediate corpus for that bridge stage.

Backward transformation rationale:
    Modern analytic languages (French, Spanish) have relatively rigid word
    order: certain bigrams (article + noun, preposition + article, etc.) are
    highly probable. Classical Latin has freer word order: any content word
    can follow any other with roughly equal probability. Mixing the bigram
    transition matrix toward uniform increases entropy in the direction of
    greater syntactic freedom.

    The mixing parameter alpha controls step size. At alpha=0.05:
        - After 10 steps: ~40% of original structure preserved
        - After 20 steps: ~36% preserved
        - After 40 steps: ~13% preserved
        - After 60 steps: ~5% preserved (near-uniform)

    The algorithm runs until fingerprint stability, not until the model
    reaches uniform. The stable point is the finding.

Vocabulary:
    The vocabulary is fixed at initialisation from the source corpus (capped
    at MAX_VOCAB = 5000 most frequent tokens, matching the co-occurrence
    module). Over iterations, the same tokens are used with different
    frequency distributions — the vocabulary does not change.

    This means generated intermediate corpora are composed of French (or
    Italian, etc.) tokens arranged in increasingly synthetic statistical
    patterns. The tokens are not Latin. The STRUCTURE is what moves toward
    Latin. This is deliberate: see docs/decisions/010_retrodiction_algorithm.md.

Sequence lengths:
    Sampled from the empirical distribution of the source corpus, preserving
    the statistical character of sentence lengths across iterations.
"""

import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

MAX_VOCAB = 5000   # must match cooccurrence.MAX_VOCAB


# ---------------------------------------------------------------------------
# Bigram language model
# ---------------------------------------------------------------------------

class BigramModel:
    """
    A bigram language model: P(token_j | token_i) for all i, j in vocabulary.

    The transition matrix is dense (V x V) with rows summing to 1.0.
    Mixing toward uniform increases entropy, corresponding to freer word order.
    """

    def __init__(
        self,
        vocab: list[str],
        token2idx: dict[str, int],
        transitions: np.ndarray,
        seq_lengths: np.ndarray,
    ) -> None:
        """
        Args:
            vocab:        List of tokens (index -> token).
            token2idx:    Dict mapping token -> index.
            transitions:  shape (V, V), transitions[i, j] = P(j | i).
            seq_lengths:  1-D array of observed sequence lengths for sampling.
        """
        self.vocab = vocab
        self.token2idx = token2idx
        self.transitions = transitions
        self.seq_lengths = seq_lengths
        self.V = len(vocab)

    @classmethod
    def from_sequences(
        cls,
        sequences: list[list[str]],
        max_vocab: int = MAX_VOCAB,
    ) -> "BigramModel":
        """
        Build a BigramModel from a list of token sequences.

        Args:
            sequences: Source corpus as list of token lists.
            max_vocab: Vocabulary size cap (most frequent tokens retained).

        Returns:
            BigramModel fitted to the corpus.
        """
        # Build vocabulary: top-max_vocab by frequency
        freq: dict[str, int] = {}
        for seq in sequences:
            for tok in seq:
                freq[tok] = freq.get(tok, 0) + 1

        sorted_tokens = sorted(freq, key=lambda t: -freq[t])[:max_vocab]
        vocab = sorted_tokens
        token2idx = {tok: i for i, tok in enumerate(vocab)}
        V = len(vocab)

        # Count bigrams (only within vocab)
        counts = np.zeros((V, V), dtype=np.float64)
        for seq in sequences:
            in_vocab = [tok for tok in seq if tok in token2idx]
            for a, b in zip(in_vocab, in_vocab[1:]):
                counts[token2idx[a], token2idx[b]] += 1.0

        # Normalise rows -> transition probabilities
        row_sums = counts.sum(axis=1, keepdims=True)
        # Rows with zero count get uniform distribution
        zero_rows = (row_sums == 0).flatten()
        row_sums[zero_rows] = 1.0
        transitions = counts / row_sums
        transitions[zero_rows] = 1.0 / V

        # Collect observed sequence lengths
        seq_lengths = np.array([len(s) for s in sequences if len(s) > 0], dtype=np.int32)

        log.info(
            "BigramModel: V=%d, non-zero bigrams=%d, seq_lengths mean=%.1f",
            V, int(np.count_nonzero(counts)), float(seq_lengths.mean()),
        )
        return cls(vocab, token2idx, transitions, seq_lengths)

    def mix_toward_uniform(self, alpha: float) -> "BigramModel":
        """
        Apply one backward transformation step.

        Mixes the transition matrix toward uniform:
            T_new[i, j] = (1 - alpha) * T[i, j] + alpha * (1/V)

        This increases bigram entropy (freer word order), moving the
        model in the direction of more synthetic grammar.

        Args:
            alpha: Mixing weight toward uniform, in (0, 1).

        Returns:
            New BigramModel with updated transitions.
        """
        uniform = np.full((self.V, self.V), 1.0 / self.V, dtype=np.float64)
        new_transitions = (1.0 - alpha) * self.transitions + alpha * uniform
        return BigramModel(self.vocab, self.token2idx, new_transitions, self.seq_lengths)

    def sample_sequence(self, length: int, rng: np.random.Generator) -> list[str]:
        """
        Sample a single token sequence of the given length.

        Starts from a random token, then follows bigram transitions.
        """
        idx = int(rng.integers(0, self.V))
        result = [self.vocab[idx]]
        for _ in range(length - 1):
            idx = int(rng.choice(self.V, p=self.transitions[idx]))
            result.append(self.vocab[idx])
        return result

    def sample_corpus(
        self,
        num_sequences: int,
        rng: np.random.Generator,
    ) -> list[list[str]]:
        """
        Sample a synthetic corpus of num_sequences sequences.

        Sequence lengths are drawn from the empirical distribution of the
        source corpus, preserving the statistical character of sentence lengths.

        Args:
            num_sequences: Number of sequences to generate.
            rng:           NumPy random generator for reproducibility.

        Returns:
            List of token sequences.
        """
        lengths = rng.choice(self.seq_lengths, size=num_sequences, replace=True)
        return [self.sample_sequence(int(length), rng) for length in lengths]

    def bigram_entropy(self) -> float:
        """
        Mean per-row Shannon entropy of the transition matrix (nats).

        A fully uniform model has entropy = log(V).
        A deterministic model has entropy = 0.
        Higher entropy = freer word order.
        """
        t = self.transitions
        # Compute t * log(t) safely: 0 * log(0) = 0 by convention
        with np.errstate(divide="ignore", invalid="ignore"):
            log_t = np.where(t > 0, np.log(t), 0.0)
        row_entropies = -(t * log_t).sum(axis=1)
        return float(row_entropies.mean())
