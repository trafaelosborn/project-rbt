"""
Incremental Scoring State
=========================
Purpose:
    Maintain running accumulators so that candidate scoring is
    O(changed_sequences × avg_seq_len) instead of O(full_corpus).

    Three acceleration layers stacked on top of each other:

    Layer 1 — score_token cache
        LatinFormReference.score_token is a pure function (deterministic given
        the static Latin reference). Adding a dict cache to LatinFormReference
        eliminates repeated per-token scoring calls inside every operator.
        This is the single largest win: operators call score_token O(100-400)
        times per attempt, and the same token forms recur across proposals.

    Layer 2 — precomputed token_counts and bigram_counts
        token_counts and bigram_counts of the current corpus are recomputed
        inside every _mutate_candidate call. Since the corpus only changes on
        acceptance (not between candidates within a proposal), these can be
        computed once per proposal and shared across all n_candidates calls.

    Layer 3 — incremental _evaluate_sequences
        The current _evaluate_sequences re-scans all 12K+ corpus tokens for
        char bigrams, char trigrams, suffixes, and word bigrams on every call.
        This class maintains those as running Counters updated only for the
        changed sequences detected by compute_sequence_delta. For a typical
        token-rewrite mutation touching 1-5% of sequences, this reduces the
        work by 20-100x.

Design constraints:
    - Correctness is mandatory. evaluate() must return the same scores as
      _evaluate_sequences() to within floating-point precision.
    - The existing python_only path is completely untouched. This is an
      opt-in acceleration layer, not a replacement.
    - No changes to operator logic, reward shaping, or coherence gating.
"""

from __future__ import annotations

import math
import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.accelerate.incremental_tensor_state import SequenceDelta, compute_sequence_delta
from src.fingerprint.ngram import build_profile
from src.retrodiction.engine_reinforced_v2 import (
    CHAR_BIGRAM_TOP_N,
    CHAR_TRIGRAM_TOP_N,
    SUFFIX_LEN,
    SUFFIX_TOP_N,
    CandidateState,
    LatinFormReference,
    _build_sparse_profile,
    _extract_char_ngrams_from_sequences,
    _extract_suffixes_from_sequences,
    _sparse_profile_cosine,
)
from src.retrodiction.engine_reinforced import LatinReference
from src.retrodiction.similarity import (
    COHERENCE_MARGIN_THRESHOLD,
    ReferenceSet,
    TOP_K,
    structural_vector,
    top_k_coverage,
)

# Optional Fortran cosine scorer — imported lazily to avoid circular deps
# and to allow graceful degradation if Fortran is unavailable.
try:
    from src.accelerate.fortran_cosine import FortranCosineScorer as _FortranCosineScorer
except ImportError:
    _FortranCosineScorer = None  # type: ignore[assignment,misc]

log = logging.getLogger(__name__)

WORD_NGRAM_TOP_N = 5000  # matches build_profile call in _evaluate_sequences


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _char_ngrams_of(tok: str, n: int) -> list[str]:
    """Return all char n-grams from the padded token `^tok$`."""
    padded = f"^{tok}$"
    if len(padded) < n:
        return []
    return [padded[i : i + n] for i in range(len(padded) - n + 1)]


def _update_char_counts(
    bg_counts: Counter,
    tg_counts: Counter,
    sfx_counts: Counter,
    seq: list[str],
    sign: int,
    suffix_len: int,
) -> None:
    """Add (sign=+1) or subtract (sign=-1) char n-grams for a sequence."""
    for tok in seq:
        padded = f"^{tok}$"
        n = len(padded)
        for i in range(n - 1):
            bg_counts[padded[i : i + 2]] += sign
        for i in range(n - 2):
            tg_counts[padded[i : i + 3]] += sign
        if len(tok) >= suffix_len:
            sfx_counts[tok[-suffix_len:]] += sign


def _update_word_bigrams(counts: Counter, seq: list[str], sign: int) -> None:
    for i in range(len(seq) - 1):
        counts[(seq[i], seq[i + 1])] += sign


def _update_word_trigrams(counts: Counter, seq: list[str], sign: int) -> None:
    for i in range(len(seq) - 2):
        counts[(seq[i], seq[i + 1], seq[i + 2])] += sign


def _clean_counter(c: Counter) -> Counter:
    """Remove zero/negative entries (left by subtraction)."""
    return Counter({k: v for k, v in c.items() if v > 0})


def _word_profile(counts: Counter) -> dict[str, float]:
    """Build a normalized word n-gram profile from a raw Counter."""
    return build_profile(counts, WORD_NGRAM_TOP_N)


def _top_k_coverage_from_counter(
    counts: Counter,
    *,
    k: int = TOP_K,
    top_n: int = WORD_NGRAM_TOP_N,
) -> float:
    """
    Compute top-k coverage directly from raw n-gram counts.

    This matches `top_k_coverage(build_profile(counts, top_n), k)` without
    constructing profile dicts or joined n-gram strings.
    """
    if not counts:
        return 0.0
    top = counts.most_common(top_n)
    if not top:
        return 0.0
    total = float(sum(count for _, count in top))
    if total <= 0.0:
        return 0.0
    covered = float(sum(count for _, count in top[:k]))
    return covered / total


# ---------------------------------------------------------------------------
# Scores dataclass returned by evaluate()
# ---------------------------------------------------------------------------

@dataclass
class CandidateScores:
    """Scoring results returned by IncrementalScoringState.evaluate()."""
    structural_vector: np.ndarray
    latin_structural_score: float
    latin_form_score: float
    form_details: dict
    total_score: float
    scores: dict
    diagnostics: dict
    type_token_ratio: float
    bigram_coverage: float
    trigram_coverage: float
    bigram_profile: dict
    trigram_profile: dict


@dataclass
class _VirtualScoringState:
    """Non-committed counter state for one virtual candidate corpus."""

    char_bg: Counter
    char_tg: Counter
    sfx: Counter
    word_bg: Counter
    word_tg: Counter
    token_counts: Counter
    total_tokens: int
    total_seq_len: int
    n_sequences: int


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class IncrementalScoringState:
    """
    Running accumulators for fast candidate scoring.

    Build once from the initial corpus sequences, then call:
        evaluate(new_sequences, mutation_cost, ...) -> CandidateScores
        commit(accepted_sequences)                  -> updates state

    The evaluate() path detects which sequences changed (via
    compute_sequence_delta), applies the delta to copies of the running
    counters, and computes scores without touching the rest of the corpus.

    Typical speedup vs full _evaluate_sequences():
        - form score: 20-100x (char ngram extraction over 1-5 seqs not 800)
        - structural score: 20-100x (word bigram/trigram over changed seqs only)
        - token_counts / bigram_counts: free (precomputed properties)
    """

    def __init__(
        self,
        sequences: list[list[str]],
        latin_form_ref: LatinFormReference,
        latin_structural_ref: LatinReference,
        references: ReferenceSet,
        suffix_len: int = SUFFIX_LEN,
        fortran_cosine_scorer=None,
    ) -> None:
        self._latin_form_ref = latin_form_ref
        self._latin_structural_ref = latin_structural_ref
        self._references = references
        self._suffix_len = suffix_len
        self._sequences: list[list[str]] = [list(s) for s in sequences]
        # Optional Fortran cosine scorer — replaces Python dict-intersection
        # cosines in _score_from_counters when available.
        self._fortran_cosine_scorer = fortran_cosine_scorer
        self._latin_reward_vec = np.asarray(
            getattr(latin_structural_ref, "reward_vec", np.asarray(latin_structural_ref.vec)[:3]),
            dtype=np.float64,
        )
        self._latin_score_scale = float(getattr(latin_structural_ref, "score_scale", 5.0))
        self._markov_vec = np.asarray(self._references.markov, dtype=np.float64)
        self._sumerian_vec = np.asarray(self._references.sumerian, dtype=np.float64)
        self._real_language_centroid = np.asarray(self._references.real_language_centroid, dtype=np.float64)
        self._real_language_scale = np.asarray(self._references.real_language_scale, dtype=np.float64)
        self._build(sequences)
        log.debug(
            "IncrementalScoringState built: %d seqs, %d tokens, %d types (fortran_cosine=%s)",
            len(sequences),
            self._total_tokens,
            self._unique_type_count,
            fortran_cosine_scorer is not None,
        )

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _build(self, sequences: list[list[str]]) -> None:
        # Token-level state
        self._token_counts: Counter = Counter(
            tok for seq in sequences for tok in seq
        )
        self._total_tokens: int = int(sum(self._token_counts.values()))
        self._unique_type_count: int = len(self._token_counts)
        self._total_seq_len: int = self._total_tokens  # same for flat seqs

        # Word n-gram state
        self._word_bigram_counts: Counter = Counter()
        self._word_trigram_counts: Counter = Counter()
        for seq in sequences:
            _update_word_bigrams(self._word_bigram_counts, seq, +1)
            _update_word_trigrams(self._word_trigram_counts, seq, +1)

        # Char n-gram state (weighted by occurrence, not type)
        self._char_bg_counts: Counter = _extract_char_ngrams_from_sequences(
            sequences, 2
        )
        self._char_tg_counts: Counter = _extract_char_ngrams_from_sequences(
            sequences, 3
        )
        self._sfx_counts: Counter = _extract_suffixes_from_sequences(
            sequences, self._suffix_len
        )

    @classmethod
    def from_sequences(
        cls,
        sequences: list[list[str]],
        latin_form_ref: LatinFormReference,
        latin_structural_ref: LatinReference,
        references: ReferenceSet,
        fortran_cosine_scorer=None,
    ) -> "IncrementalScoringState":
        return cls(sequences, latin_form_ref, latin_structural_ref, references,
                   fortran_cosine_scorer=fortran_cosine_scorer)

    # ------------------------------------------------------------------
    # Public properties (free — no recomputation)
    # ------------------------------------------------------------------

    @property
    def token_counts(self) -> Counter:
        """Committed token counts for the current corpus. No recomputation."""
        return self._token_counts

    @property
    def word_bigram_counts(self) -> Counter:
        """Committed word bigram counts. No recomputation."""
        return self._word_bigram_counts

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def evaluate(
        self,
        new_sequences: list[list[str]],
        mutation_cost: float,
        form_weight: float,
        coherence_weight: float,
        mutation_cost_weight: float,
    ) -> CandidateScores:
        """
        Score new_sequences against the committed corpus state.

        Fast path: detects changed sequences, applies delta to Counter copies,
        computes scores without scanning unchanged sequences.

        Returns the same values as _evaluate_sequences() to floating-point
        precision.
        """
        virtual = self._virtual_state_from_sequences(new_sequences)
        return self._score_virtual_state(
            virtual,
            mutation_cost,
            form_weight,
            coherence_weight,
            mutation_cost_weight,
        )

    def evaluate_changed_sequences(
        self,
        changed_sequences: dict[int, list[str]],
        mutation_cost: float,
        form_weight: float,
        coherence_weight: float,
        mutation_cost_weight: float,
    ) -> CandidateScores:
        """
        Score a sparse candidate represented only by changed sequence rows.

        This avoids building a full candidate corpus and bypasses
        compute_sequence_delta() when the mutation layer already knows exactly
        which sequence indices were touched.
        """
        virtual = self._virtual_state_from_changed_sequences(changed_sequences)
        return self._score_virtual_state(
            virtual,
            mutation_cost,
            form_weight,
            coherence_weight,
            mutation_cost_weight,
        )

    def evaluate_batch(
        self,
        candidate_sequences: list[list[list[str]]],
        mutation_costs: list[float],
        form_weight: float,
        coherence_weight: float,
        mutation_cost_weight: float,
    ) -> list[CandidateScores]:
        """
        Score N candidate corpora against the committed state.

        When a FortranCosineScorer is present, the form-score component is
        computed for the whole candidate batch in one compiled call.
        Structural/coherence math stays identical to the single-candidate path.
        """
        if len(candidate_sequences) != len(mutation_costs):
            raise ValueError(
                "candidate_sequences and mutation_costs must have the same length: "
                f"{len(candidate_sequences)} != {len(mutation_costs)}"
            )
        if not candidate_sequences:
            return []

        form_scores: np.ndarray | None = None
        if self._fortran_cosine_scorer is not None:
            virtual_states: list[_VirtualScoringState] = []
            form_deltas: list[tuple[Counter, Counter, Counter]] = []
            for sequences in candidate_sequences:
                delta = compute_sequence_delta(self._sequences, sequences)
                virtual_states.append(
                    self._virtual_state_from_sequences(
                        sequences,
                        delta=delta,
                        include_char_counts=False,
                    )
                )
                form_deltas.append(
                    self._char_counter_deltas_from_sequences(
                        sequences,
                        delta=delta,
                    )
                )
            form_scores = self._fortran_cosine_scorer.score_form_batch_from_deltas(
                self._char_bg_counts,
                self._char_tg_counts,
                self._sfx_counts,
                form_deltas,
            )
            return self._score_virtual_state_batch(
                virtual_states,
                mutation_costs,
                form_weight,
                coherence_weight,
                mutation_cost_weight,
                form_scores,
            )
        else:
            virtual_states = [
                self._virtual_state_from_sequences(sequences)
                for sequences in candidate_sequences
            ]

        scored: list[CandidateScores] = []
        for idx, (virtual, mutation_cost) in enumerate(zip(virtual_states, mutation_costs)):
            precomputed_score = None if form_scores is None else float(form_scores[idx])
            scored.append(
                self._score_virtual_state(
                    virtual,
                    mutation_cost,
                    form_weight,
                    coherence_weight,
                    mutation_cost_weight,
                    precomputed_form_score=precomputed_score,
                )
            )
        return scored

    def evaluate_batch_changed_sequences(
        self,
        batch_changed_sequences: list[dict[int, list[str]]],
        mutation_costs: list[float],
        form_weight: float,
        coherence_weight: float,
        mutation_cost_weight: float,
    ) -> list[CandidateScores]:
        """
        Batch-score candidates represented as sparse changed-sequence maps.

        This is the preferred path when the mutation layer can produce
        lightweight delta payloads instead of fully materialized corpora.
        """
        if len(batch_changed_sequences) != len(mutation_costs):
            raise ValueError(
                "batch_changed_sequences and mutation_costs must have the same length: "
                f"{len(batch_changed_sequences)} != {len(mutation_costs)}"
            )
        if not batch_changed_sequences:
            return []

        form_scores: np.ndarray | None = None
        if self._fortran_cosine_scorer is not None:
            virtual_states: list[_VirtualScoringState] = []
            form_deltas: list[tuple[Counter, Counter, Counter]] = []
            for changed_sequences in batch_changed_sequences:
                virtual_states.append(
                    self._virtual_state_from_changed_sequences(
                        changed_sequences,
                        include_char_counts=False,
                    )
                )
                form_deltas.append(
                    self._char_counter_deltas_from_changed_sequences(changed_sequences)
                )
            form_scores = self._fortran_cosine_scorer.score_form_batch_from_deltas(
                self._char_bg_counts,
                self._char_tg_counts,
                self._sfx_counts,
                form_deltas,
            )
            return self._score_virtual_state_batch(
                virtual_states,
                mutation_costs,
                form_weight,
                coherence_weight,
                mutation_cost_weight,
                form_scores,
            )

        scored: list[CandidateScores] = []
        for changed_sequences, mutation_cost in zip(batch_changed_sequences, mutation_costs):
            scored.append(
                self.evaluate_changed_sequences(
                    changed_sequences,
                    mutation_cost,
                    form_weight,
                    coherence_weight,
                    mutation_cost_weight,
                )
            )
        return scored

    def _score_virtual_state_batch(
        self,
        virtual_states: list[_VirtualScoringState],
        mutation_costs: list[float],
        form_weight: float,
        coherence_weight: float,
        mutation_cost_weight: float,
        form_scores: np.ndarray,
    ) -> list[CandidateScores]:
        """Vectorized structural/coherence scoring for the Fortran batch path."""
        m = len(virtual_states)
        ttr = np.zeros(m, dtype=np.float64)
        bg_cov = np.zeros(m, dtype=np.float64)
        tg_cov = np.zeros(m, dtype=np.float64)
        log_mean = np.zeros(m, dtype=np.float64)

        for idx, virtual in enumerate(virtual_states):
            if virtual.total_tokens > 0:
                ttr[idx] = len(virtual.token_counts) / virtual.total_tokens
            if virtual.n_sequences > 0:
                log_mean[idx] = math.log1p(virtual.total_seq_len / virtual.n_sequences)
            bg_cov[idx] = _top_k_coverage_from_counter(virtual.word_bg, k=TOP_K, top_n=WORD_NGRAM_TOP_N)
            tg_cov[idx] = _top_k_coverage_from_counter(virtual.word_tg, k=TOP_K, top_n=WORD_NGRAM_TOP_N)

        vecs = np.column_stack((ttr, bg_cov, tg_cov, log_mean)).astype(np.float64, copy=False)
        reward = vecs[:, : self._latin_reward_vec.shape[0]]
        structural_scores = -self._latin_score_scale * np.linalg.norm(
            reward - self._latin_reward_vec[None, :],
            axis=1,
        )

        vs_markov = self._batch_cosine_similarity(vecs, self._markov_vec)
        vs_sumerian = self._batch_cosine_similarity(vecs, self._sumerian_vec)
        dist_real = np.linalg.norm((vecs - self._real_language_centroid[None, :]) / self._real_language_scale[None, :], axis=1)
        dist_markov = np.linalg.norm((vecs - self._markov_vec[None, :]) / self._real_language_scale[None, :], axis=1)
        margins = dist_markov - dist_real

        scored: list[CandidateScores] = []
        for idx, (virtual, mutation_cost) in enumerate(zip(virtual_states, mutation_costs)):
            form_score = float(form_scores[idx])
            margin = float(margins[idx])
            if margin >= COHERENCE_MARGIN_THRESHOLD:
                label = "coherent"
            elif margin >= 0.0:
                label = "borderline"
            else:
                label = "noise_like"
            total_score = (
                float(structural_scores[idx])
                + form_weight * form_score
                + coherence_weight * margin
                - mutation_cost_weight * mutation_cost
            )
            scored.append(
                CandidateScores(
                    structural_vector=vecs[idx],
                    latin_structural_score=float(structural_scores[idx]),
                    latin_form_score=form_score,
                    form_details={
                        "latin_form_score": form_score,
                        "latin_char_bigram_cosine": float("nan"),
                        "latin_char_trigram_cosine": float("nan"),
                        "latin_suffix_cosine": float("nan"),
                    },
                    total_score=float(total_score),
                    scores={
                        "vs_markov_noise": float(vs_markov[idx]),
                        "vs_sumerian": float(vs_sumerian[idx]),
                        "vs_portuguese_control": None,
                        "vs_latin_ground_truth": None,
                    },
                    diagnostics={
                        "distance_to_real_language_centroid": float(dist_real[idx]),
                        "distance_to_markov_noise": float(dist_markov[idx]),
                        "language_likeness_margin": margin,
                        "coherence_label": label,
                    },
                    type_token_ratio=float(ttr[idx]),
                    bigram_coverage=float(bg_cov[idx]),
                    trigram_coverage=float(tg_cov[idx]),
                    bigram_profile={},
                    trigram_profile={},
                )
            )
        return scored

    @staticmethod
    def _batch_cosine_similarity(matrix: np.ndarray, ref: np.ndarray) -> np.ndarray:
        """Cosine similarity between each row of `matrix` and `ref`."""
        dots = matrix @ ref
        row_norms = np.linalg.norm(matrix, axis=1)
        ref_norm = float(np.linalg.norm(ref))
        denom = row_norms * ref_norm
        safe = np.where(denom > 0.0, denom, 1.0)
        return np.where(denom > 0.0, dots / safe, 0.0)

    def _virtual_state_from_sequences(
        self,
        new_sequences: list[list[str]],
        *,
        delta: SequenceDelta | None = None,
        include_char_counts: bool = True,
    ) -> _VirtualScoringState:
        """Construct a non-committed counter state for a candidate corpus."""
        delta = delta or compute_sequence_delta(self._sequences, new_sequences)

        if delta.is_noop:
            return _VirtualScoringState(
                char_bg=Counter(self._char_bg_counts) if include_char_counts else Counter(),
                char_tg=Counter(self._char_tg_counts) if include_char_counts else Counter(),
                sfx=Counter(self._sfx_counts) if include_char_counts else Counter(),
                word_bg=Counter(self._word_bigram_counts),
                word_tg=Counter(self._word_trigram_counts),
                token_counts=Counter(self._token_counts),
                total_tokens=self._total_tokens,
                total_seq_len=self._total_seq_len,
                n_sequences=len(new_sequences),
            )

        changed_idx = delta.changed_indices
        new_char_bg = Counter(self._char_bg_counts) if include_char_counts else Counter()
        new_char_tg = Counter(self._char_tg_counts) if include_char_counts else Counter()
        new_sfx = Counter(self._sfx_counts) if include_char_counts else Counter()
        new_word_bg = Counter(self._word_bigram_counts)
        new_word_tg = Counter(self._word_trigram_counts)
        new_token_counts = Counter(self._token_counts)
        new_total_seq_len = self._total_seq_len

        for idx in changed_idx:
            old_seq = self._sequences[idx]
            new_seq = new_sequences[idx]

            if include_char_counts:
                _update_char_counts(new_char_bg, new_char_tg, new_sfx, old_seq, -1, self._suffix_len)
                _update_char_counts(new_char_bg, new_char_tg, new_sfx, new_seq, +1, self._suffix_len)

            _update_word_bigrams(new_word_bg, old_seq, -1)
            _update_word_bigrams(new_word_bg, new_seq, +1)
            _update_word_trigrams(new_word_tg, old_seq, -1)
            _update_word_trigrams(new_word_tg, new_seq, +1)

            new_token_counts.subtract(old_seq)
            new_token_counts.update(new_seq)
            new_total_seq_len += len(new_seq) - len(old_seq)

        new_token_counts = _clean_counter(new_token_counts)
        new_total_tokens = int(sum(new_token_counts.values()))
        return _VirtualScoringState(
            char_bg=new_char_bg,
            char_tg=new_char_tg,
            sfx=new_sfx,
            word_bg=new_word_bg,
            word_tg=new_word_tg,
            token_counts=new_token_counts,
            total_tokens=new_total_tokens,
            total_seq_len=new_total_seq_len,
            n_sequences=len(new_sequences),
        )

    def _virtual_state_from_changed_sequences(
        self,
        changed_sequences: dict[int, list[str]],
        *,
        include_char_counts: bool = True,
    ) -> _VirtualScoringState:
        """Construct a non-committed counter state from sparse changed rows."""
        if not changed_sequences:
            return _VirtualScoringState(
                char_bg=Counter(self._char_bg_counts) if include_char_counts else Counter(),
                char_tg=Counter(self._char_tg_counts) if include_char_counts else Counter(),
                sfx=Counter(self._sfx_counts) if include_char_counts else Counter(),
                word_bg=Counter(self._word_bigram_counts),
                word_tg=Counter(self._word_trigram_counts),
                token_counts=Counter(self._token_counts),
                total_tokens=self._total_tokens,
                total_seq_len=self._total_seq_len,
                n_sequences=len(self._sequences),
            )

        new_char_bg = Counter(self._char_bg_counts) if include_char_counts else Counter()
        new_char_tg = Counter(self._char_tg_counts) if include_char_counts else Counter()
        new_sfx = Counter(self._sfx_counts) if include_char_counts else Counter()
        new_word_bg = Counter(self._word_bigram_counts)
        new_word_tg = Counter(self._word_trigram_counts)
        new_token_counts = Counter(self._token_counts)
        new_total_seq_len = self._total_seq_len

        for idx in sorted(changed_sequences):
            old_seq = self._sequences[idx]
            new_seq = changed_sequences[idx]

            if include_char_counts:
                _update_char_counts(new_char_bg, new_char_tg, new_sfx, old_seq, -1, self._suffix_len)
                _update_char_counts(new_char_bg, new_char_tg, new_sfx, new_seq, +1, self._suffix_len)

            _update_word_bigrams(new_word_bg, old_seq, -1)
            _update_word_bigrams(new_word_bg, new_seq, +1)
            _update_word_trigrams(new_word_tg, old_seq, -1)
            _update_word_trigrams(new_word_tg, new_seq, +1)

            new_token_counts.subtract(old_seq)
            new_token_counts.update(new_seq)
            new_total_seq_len += len(new_seq) - len(old_seq)

        new_token_counts = _clean_counter(new_token_counts)
        new_total_tokens = int(sum(new_token_counts.values()))
        return _VirtualScoringState(
            char_bg=new_char_bg,
            char_tg=new_char_tg,
            sfx=new_sfx,
            word_bg=new_word_bg,
            word_tg=new_word_tg,
            token_counts=new_token_counts,
            total_tokens=new_total_tokens,
            total_seq_len=new_total_seq_len,
            n_sequences=len(self._sequences),
        )

    def _char_counter_deltas_from_sequences(
        self,
        new_sequences: list[list[str]],
        *,
        delta: SequenceDelta | None = None,
    ) -> tuple[Counter, Counter, Counter]:
        """
        Build sparse char-ngram deltas relative to the committed state.

        Used by the batch Fortran path so we can score candidate form from one
        committed baseline plus per-candidate sparse updates instead of copying
        the full char Counter state N times.
        """
        delta = delta or compute_sequence_delta(self._sequences, new_sequences)
        bg_delta: Counter = Counter()
        tg_delta: Counter = Counter()
        sfx_delta: Counter = Counter()
        if delta.is_noop:
            return bg_delta, tg_delta, sfx_delta

        for idx in delta.changed_indices:
            old_seq = self._sequences[idx]
            new_seq = new_sequences[idx]
            _update_char_counts(bg_delta, tg_delta, sfx_delta, old_seq, -1, self._suffix_len)
            _update_char_counts(bg_delta, tg_delta, sfx_delta, new_seq, +1, self._suffix_len)

        return bg_delta, tg_delta, sfx_delta

    def _char_counter_deltas_from_changed_sequences(
        self,
        changed_sequences: dict[int, list[str]],
    ) -> tuple[Counter, Counter, Counter]:
        """Build sparse char-ngram deltas from sparse changed rows only."""
        bg_delta: Counter = Counter()
        tg_delta: Counter = Counter()
        sfx_delta: Counter = Counter()
        for idx in sorted(changed_sequences):
            old_seq = self._sequences[idx]
            new_seq = changed_sequences[idx]
            _update_char_counts(bg_delta, tg_delta, sfx_delta, old_seq, -1, self._suffix_len)
            _update_char_counts(bg_delta, tg_delta, sfx_delta, new_seq, +1, self._suffix_len)
        return bg_delta, tg_delta, sfx_delta

    def _score_virtual_state(
        self,
        virtual: _VirtualScoringState,
        mutation_cost: float,
        form_weight: float,
        coherence_weight: float,
        mutation_cost_weight: float,
        *,
        precomputed_form_score: float | None = None,
    ) -> CandidateScores:
        return self._score_from_counters(
            virtual.char_bg,
            virtual.char_tg,
            virtual.sfx,
            virtual.word_bg,
            virtual.word_tg,
            virtual.token_counts,
            virtual.total_tokens,
            virtual.total_seq_len,
            virtual.n_sequences,
            mutation_cost,
            form_weight,
            coherence_weight,
            mutation_cost_weight,
            precomputed_form_score=precomputed_form_score,
        )

    def _score_from_counters(
        self,
        char_bg: Counter,
        char_tg: Counter,
        sfx: Counter,
        word_bg: Counter,
        word_tg: Counter,
        token_counts: Counter,
        total_tokens: int,
        total_seq_len: int,
        n_sequences: int,
        mutation_cost: float,
        form_weight: float,
        coherence_weight: float,
        mutation_cost_weight: float,
        precomputed_form_score: float | None = None,
    ) -> CandidateScores:
        """Compute scores from pre-built (possibly virtual) counters."""
        # --- Latin form score ---
        if precomputed_form_score is not None:
            form_score = float(precomputed_form_score)
            char_bg_cos = char_tg_cos = sfx_cos = float("nan")
            form_details = {
                "latin_form_score": float(form_score),
                "latin_char_bigram_cosine": char_bg_cos,
                "latin_char_trigram_cosine": char_tg_cos,
                "latin_suffix_cosine": sfx_cos,
            }
        elif self._fortran_cosine_scorer is not None:
            # Fortran/numpy BLAS path: dense float32 dot products against
            # pre-indexed reference arrays. Replaces 3 Python dict-intersection
            # cosines with one score_single_form call.
            form_score = self._fortran_cosine_scorer.score_single_form(
                char_bg, char_tg, sfx,
                bg_top_n=CHAR_BIGRAM_TOP_N,
                tg_top_n=CHAR_TRIGRAM_TOP_N,
                sfx_top_n=SUFFIX_TOP_N,
            )
            # Individual cosines not recomputed in the fast path; use
            # form_score itself for the detail fields (avoids re-entering Python).
            char_bg_cos = char_tg_cos = sfx_cos = float("nan")
            form_details = {
                "latin_form_score": float(form_score),
                "latin_char_bigram_cosine": char_bg_cos,
                "latin_char_trigram_cosine": char_tg_cos,
                "latin_suffix_cosine": sfx_cos,
            }
        else:
            # Python reference path (dict intersection cosine)
            bg_profile = _build_sparse_profile(char_bg, CHAR_BIGRAM_TOP_N)
            tg_profile = _build_sparse_profile(char_tg, CHAR_TRIGRAM_TOP_N)
            sfx_profile = _build_sparse_profile(sfx, SUFFIX_TOP_N)

            char_bg_cos = _sparse_profile_cosine(bg_profile, self._latin_form_ref.char_bigram_profile)
            char_tg_cos = _sparse_profile_cosine(tg_profile, self._latin_form_ref.char_trigram_profile)
            sfx_cos = _sparse_profile_cosine(sfx_profile, self._latin_form_ref.suffix_profile)
            form_score = (
                self._latin_form_ref.char_bigram_weight * char_bg_cos
                + self._latin_form_ref.char_trigram_weight * char_tg_cos
                + self._latin_form_ref.suffix_weight * sfx_cos
            )

            form_details = {
                "latin_form_score": float(form_score),
                "latin_char_bigram_cosine": float(char_bg_cos),
                "latin_char_trigram_cosine": float(char_tg_cos),
                "latin_suffix_cosine": float(sfx_cos),
            }

        # --- Structural score ---
        ttr = len(token_counts) / total_tokens if total_tokens > 0 else 0.0
        mean_seq_len = total_seq_len / n_sequences if n_sequences > 0 else 0.0

        word_bg_profile = _word_profile(word_bg)
        word_tg_profile = _word_profile(word_tg)

        bg_cov = top_k_coverage(word_bg_profile, TOP_K)
        tg_cov = top_k_coverage(word_tg_profile, TOP_K)

        vec = np.array(
            [ttr, bg_cov, tg_cov, math.log1p(mean_seq_len)],
            dtype=np.float64,
        )

        structural_score = self._latin_structural_ref.score(vec)
        # Use the pre-built vec directly instead of re-deriving it inside score()
        from src.retrodiction.similarity import cosine_similarity
        scores = {
            "vs_markov_noise": cosine_similarity(vec, self._references.markov),
            "vs_sumerian": cosine_similarity(vec, self._references.sumerian),
            "vs_portuguese_control": None,
            "vs_latin_ground_truth": None,
        }
        diagnostics = self._references.coherence_from_vector(vec)

        total_score = (
            structural_score
            + form_weight * form_score
            + coherence_weight * diagnostics["language_likeness_margin"]
            - mutation_cost_weight * mutation_cost
        )

        return CandidateScores(
            structural_vector=vec,
            latin_structural_score=float(structural_score),
            latin_form_score=float(form_score),
            form_details=form_details,
            total_score=float(total_score),
            scores=scores,
            diagnostics=diagnostics,
            type_token_ratio=float(ttr),
            bigram_coverage=float(bg_cov),
            trigram_coverage=float(tg_cov),
            bigram_profile=word_bg_profile,
            trigram_profile=word_tg_profile,
        )

    # ------------------------------------------------------------------
    # Commit
    # ------------------------------------------------------------------

    def commit(self, new_sequences: list[list[str]]) -> None:
        """
        Commit an accepted mutation. Updates all running state in-place.
        Must be called exactly once per accepted stage, after evaluate().
        """
        delta = compute_sequence_delta(self._sequences, new_sequences)
        if delta.is_noop:
            return

        changed_idx = delta.changed_indices

        for idx in changed_idx:
            old_seq = self._sequences[idx]
            new_seq = new_sequences[idx]

            _update_char_counts(
                self._char_bg_counts, self._char_tg_counts, self._sfx_counts,
                old_seq, -1, self._suffix_len,
            )
            _update_char_counts(
                self._char_bg_counts, self._char_tg_counts, self._sfx_counts,
                new_seq, +1, self._suffix_len,
            )

            _update_word_bigrams(self._word_bigram_counts, old_seq, -1)
            _update_word_bigrams(self._word_bigram_counts, new_seq, +1)
            _update_word_trigrams(self._word_trigram_counts, old_seq, -1)
            _update_word_trigrams(self._word_trigram_counts, new_seq, +1)

            self._token_counts.subtract(old_seq)
            self._token_counts.update(new_seq)
            self._total_seq_len += len(new_seq) - len(old_seq)

            self._sequences[idx] = list(new_seq)

        # Prune zeros
        self._token_counts = _clean_counter(self._token_counts)
        self._total_tokens = int(sum(self._token_counts.values()))
        self._unique_type_count = len(self._token_counts)

        log.debug(
            "Committed delta: %d sequences changed, %d tokens now",
            len(changed_idx),
            self._total_tokens,
        )


__all__ = [
    "CandidateScores",
    "IncrementalScoringState",
]
