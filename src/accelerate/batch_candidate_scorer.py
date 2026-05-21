"""
Batch Candidate Scorer
=======================
Purpose:
    Define the interface contract for batched candidate evaluation and provide
    a Python reference implementation.

Status
------
The Python reference implementation (PythonBatchCandidateScorer) is
production-correct but not accelerated — it calls _evaluate_sequences()
once per candidate in a sequential loop, identical to what v4 does today.

The interface (BatchCandidateScorer protocol) documents exactly what a
Fortran or numpy-parallel implementation must satisfy to drop in as a
replacement. See docs/decisions/028_v5_controller_architecture.md for the
full gap analysis.

When to use this
----------------
The reference scorer is useful for:
  1. Benchmarking throughput across candidate counts (the bottleneck is
     candidate scoring, and this makes that explicit)
  2. Establishing a correctness baseline before a Fortran implementation
  3. Enabling the v5 engine to be written against the abstract interface

The Fortran scorer is NOT yet implemented. Do not claim otherwise.

Interface contract
------------------
Any scorer (Python reference, numpy batch, Fortran batch) must satisfy:

    score_batch(
        current_sequences: list[list[str]],
        candidate_sequences: list[list[list[str]]],
        mutation_costs: list[float],
        form_weight: float,
        coherence_weight: float,
        mutation_cost_weight: float,
        transparency_scorer: SemanticTransparencyScorer | None,
        transparency_weight: float,
    ) -> list[CandidateResult]

where CandidateResult contains (total_score, latin_structural_score,
latin_form_score, diagnostics).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from src.accelerate.semantic_transparency import SemanticTransparencyScorer
    from src.retrodiction.engine_reinforced import LatinReference
    from src.retrodiction.engine_reinforced_v2 import LatinFormReference
    from src.retrodiction.similarity import ReferenceSet

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class CandidateResult:
    """
    Score result for one candidate from score_batch().
    Carries only what the acceptance loop needs.
    """
    candidate_index: int
    total_score: float
    latin_structural_score: float
    latin_form_score: float
    transparency_score: float | None
    diagnostics: dict


# ---------------------------------------------------------------------------
# Protocol (the interface contract)
# ---------------------------------------------------------------------------

@runtime_checkable
class BatchCandidateScorer(Protocol):
    """
    Interface contract for batched candidate evaluation.

    Any implementation (Python reference, numpy, Fortran) must satisfy this.
    The engine calls score_batch() once per proposal with all N candidate
    mutations and receives N CandidateResult objects back.

    Implementations must be deterministic given the same inputs.
    They must return results in the same order as `candidate_sequences`.
    """

    def score_batch(
        self,
        candidate_sequences: list[list[list[str]]],
        mutation_costs: list[float],
        form_weight: float,
        coherence_weight: float,
        mutation_cost_weight: float,
        transparency_scorer: "SemanticTransparencyScorer | None" = None,
        transparency_weight: float = 0.0,
    ) -> list[CandidateResult]:
        """
        Score N candidate corpora.

        Parameters
        ----------
        candidate_sequences : list of N corpora, each a list of sequences
        mutation_costs      : list of N mutation cost scalars
        form_weight         : weight for the Latin form term
        coherence_weight    : weight for the coherence term
        mutation_cost_weight: weight for the mutation cost penalty
        transparency_scorer : optional transparency scorer (None = disabled)
        transparency_weight : weight for the transparency term

        Returns
        -------
        list of CandidateResult, one per candidate, in input order
        """
        ...


# ---------------------------------------------------------------------------
# Python reference implementation
# ---------------------------------------------------------------------------

class PythonBatchCandidateScorer:
    """
    Python reference implementation of BatchCandidateScorer.

    Calls the engine's _evaluate_sequences equivalent for each candidate
    in a sequential loop. Functionally identical to the current v4 loop —
    this implementation makes the interface explicit without adding
    acceleration.

    Use for:
    - Correctness baseline
    - Benchmarking to measure what acceleration is worth
    - Drop-in until a parallel implementation exists

    Not for:
    - Claiming Fortran batch is implemented
    - Production runs expecting acceleration beyond current v4
    """

    def __init__(
        self,
        latin_structural_ref: "LatinReference",
        latin_form_ref: "LatinFormReference",
        references: "ReferenceSet",
    ) -> None:
        self._structural_ref = latin_structural_ref
        self._form_ref = latin_form_ref
        self._references = references

    def score_batch(
        self,
        candidate_sequences: list[list[list[str]]],
        mutation_costs: list[float],
        form_weight: float,
        coherence_weight: float,
        mutation_cost_weight: float,
        transparency_scorer: "SemanticTransparencyScorer | None" = None,
        transparency_weight: float = 0.0,
    ) -> list[CandidateResult]:
        """Sequential Python loop — reference correctness, no acceleration."""
        from src.fingerprint.ngram import build_profile, extract_ngrams
        from src.retrodiction.similarity import structural_vector

        results: list[CandidateResult] = []

        for i, (sequences, mutation_cost) in enumerate(
            zip(candidate_sequences, mutation_costs)
        ):
            bg_counts = extract_ngrams(sequences, 2)
            tg_counts = extract_ngrams(sequences, 3)
            bg_profile = build_profile(bg_counts, 5000)
            tg_profile = build_profile(tg_counts, 5000)
            vec = structural_vector(sequences, bg_profile, tg_profile)

            structural_score = float(self._structural_ref.score(vec))
            form_details = self._form_ref.score(sequences)
            form_score = float(form_details["latin_form_score"])
            coherence = self._references.coherence_from_vector(vec)
            margin = float(coherence.get("language_likeness_margin", 0.0))

            total_score = (
                structural_score
                + form_weight * form_score
                + coherence_weight * margin
                - mutation_cost_weight * mutation_cost
            )

            t_score: float | None = None
            if transparency_scorer is not None and transparency_weight > 0.0:
                t_score = float(transparency_scorer.score(sequences))
                total_score += transparency_weight * t_score

            results.append(CandidateResult(
                candidate_index=i,
                total_score=float(total_score),
                latin_structural_score=structural_score,
                latin_form_score=form_score,
                transparency_score=t_score,
                diagnostics={
                    "form_details": form_details,
                    "coherence": coherence,
                },
            ))

        return results
