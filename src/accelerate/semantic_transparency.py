"""
Semantic Transparency Scorer
=============================
Purpose:
    Score a corpus on whether its high-frequency tokens carry recoverable
    Latin-derived semantic content, as opposed to being statistically empty
    function-word glue.

Methodological status
---------------------
This is an experimental condition, not a neutral quality measure.

The form score (already in v4) measures how well the *aggregate*
character n-gram distribution of the corpus resembles Latin. That is
a population-level signal.

Transparency measures something orthogonal: in natural language, the
most frequent words tend to be either function words (semantically empty,
high frequency) or high-utility content words (semantically rich). In
reconstructed Latin, we want the most frequent tokens to have Latin-like
morphological structure — not random filler that happens to produce a
good aggregate n-gram distribution.

Definition
----------
For each corpus:
1. Compute token frequencies.
2. Score each token individually using LatinFormReference.score_token()
   (char bigram + trigram + suffix cosine vs Latin).
3. Compute a frequency-weighted average of the top-N most frequent tokens.
4. This is the transparency score: [0, 1].

Properties
----------
- Independent of aggregate n-gram distribution (the form score measures that)
- Penalises corpora whose most common tokens are low-scoring filler
- Rewards corpora where high-frequency tokens look Latin-derived
- Calibration target: 0.5 is the natural midpoint; probe run determines
  what weight to attach to this term in the total score

Integration
-----------
    transparency_score = SemanticTransparencyScorer.score(sequences)
    total_score += transparency_weight * transparency_score

The transparency_weight must be calibrated before production use.
See docs/decisions/029_semantic_transparency_scorer.md.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.retrodiction.engine_reinforced_v2 import LatinFormReference

log = logging.getLogger(__name__)

# Number of most-frequent tokens to include in the transparency score.
# Captures the "glue word" problem: the top-50 tokens in a corpus
# account for ~60-70% of all token occurrences.
TRANSPARENCY_TOP_N = 50


@dataclass
class TransparencyResult:
    """Full breakdown of a transparency evaluation."""
    score: float                  # overall weighted average [0, 1]
    top_n: int                    # how many tokens were scored
    mean_token_score: float       # unweighted mean over top-N tokens
    freq_weighted_score: float    # frequency-weighted mean (== score)
    high_score_freq_mass: float   # fraction of top-N tokens with score >= 0.5
    token_breakdown: list[tuple[str, float, float]]  # (token, freq, score)


class SemanticTransparencyScorer:
    """
    Frequency-weighted transparency scorer.

    Build once (lightweight — just holds the reference), call score() per
    candidate corpus.
    """

    def __init__(
        self,
        latin_form_ref: "LatinFormReference",
        top_n: int = TRANSPARENCY_TOP_N,
    ) -> None:
        self._form_ref = latin_form_ref
        self._top_n = top_n

    def score(self, sequences: list[list[str]]) -> float:
        """Return the scalar transparency score for a corpus [0, 1]."""
        return self.score_full(sequences).score

    def score_full(self, sequences: list[list[str]]) -> TransparencyResult:
        """Return a full breakdown of the transparency evaluation."""
        token_counts: Counter = Counter(
            tok for seq in sequences for tok in seq
        )
        total_tokens = max(sum(token_counts.values()), 1)
        top_tokens = token_counts.most_common(self._top_n)

        if not top_tokens:
            return TransparencyResult(
                score=0.0,
                top_n=0,
                mean_token_score=0.0,
                freq_weighted_score=0.0,
                high_score_freq_mass=0.0,
                token_breakdown=[],
            )

        top_freq_total = sum(count for _, count in top_tokens)

        breakdown: list[tuple[str, float, float]] = []
        freq_weighted_sum = 0.0
        unweighted_sum = 0.0
        high_score_count = 0

        for token, count in top_tokens:
            tok_score = float(self._form_ref.score_token(token))
            freq = count / total_tokens
            freq_weighted_sum += freq * tok_score
            unweighted_sum += tok_score
            if tok_score >= 0.5:
                high_score_count += 1
            breakdown.append((token, freq, tok_score))

        # Normalize: freq_weighted_sum is relative to total corpus,
        # but we want a score in [0, 1] comparable across corpus sizes.
        # Divide by the combined frequency mass of the top-N tokens.
        top_freq_mass = top_freq_total / total_tokens
        freq_weighted_score = (freq_weighted_sum / top_freq_mass) if top_freq_mass > 0 else 0.0
        mean_token_score = unweighted_sum / len(top_tokens)
        high_score_freq_mass = high_score_count / len(top_tokens)

        return TransparencyResult(
            score=float(min(max(freq_weighted_score, 0.0), 1.0)),
            top_n=len(top_tokens),
            mean_token_score=float(mean_token_score),
            freq_weighted_score=float(freq_weighted_score),
            high_score_freq_mass=float(high_score_freq_mass),
            token_breakdown=breakdown,
        )

    @classmethod
    def from_form_ref(
        cls,
        latin_form_ref: "LatinFormReference",
        top_n: int = TRANSPARENCY_TOP_N,
    ) -> "SemanticTransparencyScorer":
        return cls(latin_form_ref, top_n=top_n)
