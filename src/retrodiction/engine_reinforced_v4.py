"""
Relational Reinforced Retrodiction V4
=====================================
Purpose:
    Phase 2 of the proposed v4 direction.

    Keep the v3 mutation and reward stack, but use Hungarian family alignment to
    schedule operator weights dynamically via an inverse-log weirdness curve.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.accelerate.incremental_scoring_state import CandidateScores, IncrementalScoringState
from src.accelerate.semantic_transparency import SemanticTransparencyScorer

# Fortran cosine — imported here so build errors surface at import time only
# when the caller opts in (use_fortran_cosine=True). If unavailable the engine
# falls back to the Python cosine path transparently.
try:
    from src.accelerate.fortran_cosine import FortranCosineScorer as _FortranCosineScorer
except ImportError:
    _FortranCosineScorer = None  # type: ignore[assignment,misc]
from src.accelerate.v4_batch_guidance import (
    BatchGuidance,
    BatchGuidanceConfig,
    TensorBatchGuidanceBuilder,
)
from src.retrodiction.engine_reinforced import LANG_CODES, LatinReference
from src.retrodiction.engine_reinforced_v2 import CandidateState, LatinFormReference, ReinforcedV2StageRecord
from src.retrodiction.engine_reinforced_v3 import (
    ReinforcedV3Config,
    RelationalReinforcedRetrodictionEngineV3,
    V3_OPERATOR_NAMES,
)
from src.retrodiction.similarity import ReferenceSet
from src.validation.hungarian_alignment import (
    FamilyAlignmentConfig,
    FamilyInventory,
    extract_family_inventory,
    hungarian_alignment_diagnostics,
    load_latin_family_reference,
)

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RETRODICTION_DIR = PROJECT_ROOT / "data" / "retrodiction"

V4_OPERATOR_NAMES = V3_OPERATOR_NAMES
WEIRD_OPERATOR_NAMES = {
    "sequence_span_rewrite",
    "function_word_burst",
    "paradigm_family_rewrite",
    "macro_bundle_rewrite",
}

V4_ALIGNMENT_UNLOCK_REASON = (
    "Phase 3 reinforcement retrodiction v4: Latin family inventory is loaded "
    "for alignment-driven operator scheduling in a target-conditioned search."
)


@dataclass
class ReinforcedV4Config(ReinforcedV3Config):
    """v4 configuration: v3 plus alignment-driven operator scheduling."""

    alignment_config: FamilyAlignmentConfig = field(default_factory=FamilyAlignmentConfig)

    acceleration_mode: str = "python_only"
    acceleration_top_k: int = 512
    acceleration_max_assignments: int = 24
    acceleration_hotspot_token_limit: int = 48
    acceleration_hotspot_pair_limit: int = 24
    acceleration_force_rebuild: bool = False
    acceleration_build_dir: str | None = None

    alignment_beta: float = 8.0
    weirdness_floor: float = 0.15
    weirdness_ceiling: float = 1.0
    weird_operator_gain: float = 1.8
    stable_operator_gain: float = 1.4

    use_incremental_scoring: bool = True
    use_fortran_cosine: bool = False   # Fortran/BLAS dense cosine in incremental path
    use_fortran_batch: bool = False    # Batch form scoring over proposal candidates
    save_dense_matrices: bool = True

    # Semantic transparency (experimental condition — off by default)
    use_semantic_transparency: bool = False
    transparency_weight: float = 0.0
    live_event_mode: str = "all"
    live_event_buffer_size: int = 64

    max_proposals: int = 140
    patience: int = 24

    def to_dict(self) -> dict:
        data = super().to_dict()
        data.update(
            {
                "alignment_config": self.alignment_config.to_dict(),
                "acceleration_mode": self.acceleration_mode,
                "acceleration_top_k": self.acceleration_top_k,
                "acceleration_max_assignments": self.acceleration_max_assignments,
                "acceleration_hotspot_token_limit": self.acceleration_hotspot_token_limit,
                "acceleration_hotspot_pair_limit": self.acceleration_hotspot_pair_limit,
                "acceleration_force_rebuild": self.acceleration_force_rebuild,
                "acceleration_build_dir": self.acceleration_build_dir,
                "alignment_beta": self.alignment_beta,
                "weirdness_floor": self.weirdness_floor,
                "weirdness_ceiling": self.weirdness_ceiling,
                "weird_operator_gain": self.weird_operator_gain,
                "stable_operator_gain": self.stable_operator_gain,
                "use_incremental_scoring": self.use_incremental_scoring,
                "use_fortran_cosine": self.use_fortran_cosine,
                "use_fortran_batch": self.use_fortran_batch,
                "save_dense_matrices": self.save_dense_matrices,
                "use_semantic_transparency": self.use_semantic_transparency,
                "transparency_weight": self.transparency_weight,
                "live_event_mode": self.live_event_mode,
                "live_event_buffer_size": self.live_event_buffer_size,
            }
        )
        return data


@dataclass
class MutationPayload:
    """
    Candidate mutation payload.

    `changed_sequences` is the sparse v5.1 path: only touched row indices are
    carried through scoring, and the full corpus is materialized only if the
    candidate is actually accepted.
    """

    sequences: list[list[str]] | None = None
    changed_sequences: dict[int, list[str]] | None = None

    @property
    def is_sparse(self) -> bool:
        return self.changed_sequences is not None and self.sequences is None

    def materialize(self, base_sequences: list[list[str]]) -> list[list[str]]:
        if self.sequences is not None:
            return self.sequences
        if not self.changed_sequences:
            self.sequences = [list(seq) for seq in base_sequences]
            return self.sequences
        materialized = list(base_sequences)
        for idx, seq in self.changed_sequences.items():
            materialized[idx] = list(seq)
        self.sequences = materialized
        return materialized


class RelationalReinforcedRetrodictionEngineV4(RelationalReinforcedRetrodictionEngineV3):
    """
    v4 uses the v3 mutation/reward stack but schedules operator weights from
    Hungarian family alignment against Latin.
    """

    def __init__(
        self,
        language: str,
        source_sequences: list[list[str]],
        latin_structural_ref: LatinReference,
        latin_form_ref: LatinFormReference,
        config: ReinforcedV4Config | None = None,
        output_dir: Path | None = None,
        references: ReferenceSet | None = None,
        family_reference_inventory: FamilyInventory | None = None,
        batch_guidance_builder: TensorBatchGuidanceBuilder | None = None,
    ) -> None:
        cfg = config or ReinforcedV4Config()
        if output_dir is None:
            output_dir = RETRODICTION_DIR / language / "v4"
        super().__init__(
            language=language,
            source_sequences=source_sequences,
            latin_structural_ref=latin_structural_ref,
            latin_form_ref=latin_form_ref,
            config=cfg,
            output_dir=output_dir,
            references=references,
        )
        self.config = cfg
        self.family_reference_inventory = family_reference_inventory or load_latin_family_reference(
            cfg.alignment_config,
            unlock_reason=V4_ALIGNMENT_UNLOCK_REASON,
        )
        self._current_operator_weights = self._normalize_weights(np.array(cfg.operator_weights, dtype=np.float64))
        self._batch_guidance_builder = batch_guidance_builder or self._default_batch_guidance_builder()
        self._scoring_state: IncrementalScoringState | None = None
        self._live_events_path = self.output_dir / "live_events.jsonl"
        self._live_event_buffer: list[dict] = []
        self._transparency_scorer: SemanticTransparencyScorer | None = (
            SemanticTransparencyScorer.from_form_ref(latin_form_ref)
            if cfg.use_semantic_transparency
            else None
        )

        # Fortran cosine scorer — built here so it is ready before run()
        self._fortran_cosine_scorer = None
        if (cfg.use_fortran_cosine or cfg.use_fortran_batch) and _FortranCosineScorer is not None:
            try:
                self._fortran_cosine_scorer = _FortranCosineScorer.build(latin_form_ref)
                log.info(
                    "FortranCosineScorer built (using_fortran=%s)",
                    self._fortran_cosine_scorer.using_fortran,
                )
            except Exception as exc:
                log.warning("FortranCosineScorer build failed (%s); will use Python cosine.", exc)

    def _evaluate_sequences(
        self,
        sequences: list[list[str]],
        mutation_cost: float,
    ):
        candidate = super()._evaluate_sequences(sequences, mutation_cost)
        if self._transparency_scorer is not None and self.config.transparency_weight > 0.0:
            t_score = self._transparency_scorer.score(sequences)
            candidate.total_score += self.config.transparency_weight * t_score
            candidate.diagnostics["transparency_score"] = float(t_score)
        return candidate

    def _stage_id(self, iteration: int) -> str:
        return f"{self.lang_code}_v4_{iteration:03d}"

    def _default_batch_guidance_builder(self) -> TensorBatchGuidanceBuilder | None:
        mode = self.config.acceleration_mode
        if mode == "python_only":
            return None
        backend_map = {
            "numpy_batch": "numpy",
            "fortran_batch": "fortran",
            "auto_batch": "auto",
        }
        if mode not in backend_map:
            raise ValueError(
                "acceleration_mode must be one of "
                "'python_only', 'numpy_batch', 'fortran_batch', or 'auto_batch', "
                f"got {mode!r}"
            )
        guidance_cfg = BatchGuidanceConfig(
            backend=backend_map[mode],
            top_k=self.config.acceleration_top_k,
            max_assignments=self.config.acceleration_max_assignments,
            hotspot_token_limit=self.config.acceleration_hotspot_token_limit,
            hotspot_pair_limit=self.config.acceleration_hotspot_pair_limit,
            build_dir=self.config.acceleration_build_dir,
            force_rebuild=self.config.acceleration_force_rebuild,
        )
        return TensorBatchGuidanceBuilder(guidance_cfg, reference_label="latin")

    def _normalize_weights(self, weights: np.ndarray) -> np.ndarray:
        weights = np.asarray(weights, dtype=np.float64)
        total = float(weights.sum())
        if total <= 0.0:
            return np.full(len(V4_OPERATOR_NAMES), 1.0 / len(V4_OPERATOR_NAMES), dtype=np.float64)
        return weights / total

    def _weirdness_from_alignment(self, alignment_score: float) -> float:
        alignment = float(min(max(alignment_score, 0.0), 1.0))
        beta = max(self.config.alignment_beta, 1e-6)
        cooldown = np.log1p(beta * alignment) / np.log1p(beta)
        weirdness = self.config.weirdness_floor + (
            self.config.weirdness_ceiling - self.config.weirdness_floor
        ) * (1.0 - cooldown)
        return float(min(max(weirdness, self.config.weirdness_floor), self.config.weirdness_ceiling))

    def _scheduled_operator_weights(self, alignment_score: float) -> np.ndarray:
        weirdness = self._weirdness_from_alignment(alignment_score)
        weights = np.array(self.config.operator_weights, dtype=np.float64)
        for i, operator in enumerate(V4_OPERATOR_NAMES):
            if operator in WEIRD_OPERATOR_NAMES:
                factor = 0.5 + self.config.weird_operator_gain * weirdness
            else:
                factor = 0.5 + self.config.stable_operator_gain * (1.0 - weirdness)
            weights[i] *= factor
        return self._normalize_weights(weights)

    def _selection_schedule_dict(self, weights: np.ndarray) -> dict[str, float]:
        return {
            operator: round(float(weight), 6)
            for operator, weight in zip(V4_OPERATOR_NAMES, weights.tolist())
        }

    def _annotate_alignment(self, candidate) -> None:
        inventory = extract_family_inventory(
            label=self.language,
            sequences=candidate.sequences,
            config=self.config.alignment_config,
        )
        alignment = hungarian_alignment_diagnostics(
            inventory=inventory,
            reference_inventory=self.family_reference_inventory,
            config=self.config.alignment_config,
        )
        weirdness = self._weirdness_from_alignment(alignment["family_alignment_score"])
        weights = self._scheduled_operator_weights(alignment["family_alignment_score"])
        self._current_operator_weights = weights
        candidate.diagnostics = {
            **candidate.diagnostics,
            "family_alignment_score": alignment["family_alignment_score"],
            "family_alignment_cost": alignment["family_alignment_cost"],
            "family_alignment_matched_family_count": alignment["matched_family_count"],
            "family_alignment_unmatched_bridge_families": alignment["unmatched_bridge_families"],
            "family_alignment_unmatched_reference_families": alignment["unmatched_reference_families"],
            "weirdness_level": weirdness,
            "scheduled_operator_weights": self._selection_schedule_dict(weights),
        }

    def _choose_operator(self, rng: np.random.Generator) -> str:
        weights = self._normalize_weights(self._current_operator_weights)
        idx = int(rng.choice(len(V4_OPERATOR_NAMES), p=weights))
        return V4_OPERATOR_NAMES[idx]

    def _initialize_batch_guidance_state(self, sequences: list[list[str]]):
        if self._batch_guidance_builder is None:
            return None
        build_initial_state = getattr(self._batch_guidance_builder, "build_initial_state", None)
        if build_initial_state is None:
            return None
        return build_initial_state(sequences)

    def _build_batch_guidance(self, sequences: list[list[str]], tensor_state=None) -> BatchGuidance | None:
        if self._batch_guidance_builder is None:
            return None
        if tensor_state is not None:
            build_from_state = getattr(self._batch_guidance_builder, "build_from_state", None)
            if build_from_state is not None:
                return build_from_state(tensor_state)
        return self._batch_guidance_builder.build(sequences)

    def _guided_token_candidates(
        self,
        token_counts: Counter,
        guidance: BatchGuidance,
        *,
        min_len: int = 0,
        max_len: int | None = None,
    ) -> list[tuple[str, int, float]]:
        candidates: list[tuple[str, int, float]] = []
        for token, guide_weight in guidance.hotspot_token_weights:
            count = int(token_counts.get(token, 0))
            if count <= 0:
                continue
            if len(token) < min_len:
                continue
            if max_len is not None and len(token) > max_len:
                continue
            candidates.append((token, count, float(guide_weight)))
        return candidates

    def _sample_guided_token(
        self,
        candidates: list[tuple[str, int, float]],
        rng: np.random.Generator,
    ) -> str | None:
        if not candidates:
            return None
        weights = np.array(
            [max(count, 1) * max(guide_weight, 1e-6) for _, count, guide_weight in candidates],
            dtype=np.float64,
        )
        weights /= weights.sum()
        idx = int(rng.choice(len(candidates), p=weights))
        return candidates[idx][0]

    def _mutate_token_char_edit_guided(
        self,
        sequences: list[list[str]],
        token_counts: Counter,
        guidance: BatchGuidance,
        rng: np.random.Generator,
    ) -> tuple[list[list[str]] | None, str, float]:
        guided = self._guided_token_candidates(token_counts, guidance, min_len=3)
        tok = self._sample_guided_token(guided, rng)
        if tok is None:
            return super()._mutate_token_char_edit(sequences, token_counts, rng)

        new_tok = self._edit_token_form(tok, rng)
        if new_tok == tok:
            return super()._mutate_token_char_edit(sequences, token_counts, rng)

        new_sequences, affected = self._apply_token_rewrite(sequences, {tok: new_tok})
        if affected == 0:
            return super()._mutate_token_char_edit(sequences, token_counts, rng)
        return new_sequences, f"guided {tok} -> {new_tok} ({affected} occurrences)", 0.25

    def _mutate_token_char_edit_guided_sparse(
        self,
        sequences: list[list[str]],
        token_counts: Counter,
        guidance: BatchGuidance,
        rng: np.random.Generator,
    ) -> tuple[MutationPayload | None, str, float]:
        guided = self._guided_token_candidates(token_counts, guidance, min_len=3)
        tok = self._sample_guided_token(guided, rng)
        if tok is None:
            return self._mutate_token_char_edit_sparse(sequences, token_counts, rng)

        new_tok = self._edit_token_form(tok, rng)
        if new_tok == tok:
            return self._mutate_token_char_edit_sparse(sequences, token_counts, rng)

        changed_sequences, affected = self._apply_token_rewrite_sparse(sequences, {tok: new_tok})
        if affected == 0:
            return self._mutate_token_char_edit_sparse(sequences, token_counts, rng)
        return MutationPayload(changed_sequences=changed_sequences), f"guided {tok} -> {new_tok} ({affected} occurrences)", 0.25

    def _mutate_suffix_family_guided(
        self,
        sequences: list[list[str]],
        token_counts: Counter,
        guidance: BatchGuidance,
        rng: np.random.Generator,
    ) -> tuple[list[list[str]] | None, str, float]:
        guided = self._guided_token_candidates(token_counts, guidance, min_len=4)
        target = self._sample_guided_token(guided, rng)
        if target is None:
            return super()._mutate_suffix_family(sequences, token_counts, rng)

        suffix_len = int(rng.choice([2, 3, 4], p=[0.35, 0.45, 0.20]))
        suffix_len = min(suffix_len, len(target) - 1)
        suffix = target[-suffix_len:]
        family = [tok for tok in token_counts if tok.endswith(suffix) and len(tok) > suffix_len]
        if len(family) < 2:
            return super()._mutate_suffix_family(sequences, token_counts, rng)

        base_score = sum(token_counts[tok] * self.latin_form_ref.score_token(tok) for tok in family)
        best_suffix = suffix
        best_score = base_score

        for _ in range(self.config.suffix_candidate_samples):
            candidate_suffix = self.latin_form_ref.sample_suffix(rng)
            if candidate_suffix == suffix:
                candidate_suffix = self._random_edit_token_form(suffix, rng)
            if candidate_suffix == suffix or len(candidate_suffix) < 1:
                continue
            candidate_score = sum(
                token_counts[tok] * self.latin_form_ref.score_token(tok[:-suffix_len] + candidate_suffix)
                for tok in family
            )
            if candidate_score > best_score:
                best_suffix = candidate_suffix
                best_score = candidate_score

        if best_suffix == suffix:
            return super()._mutate_suffix_family(sequences, token_counts, rng)

        rewrite_map = {tok: tok[:-suffix_len] + best_suffix for tok in family}
        new_sequences, affected = self._apply_token_rewrite(sequences, rewrite_map)
        if affected == 0:
            return super()._mutate_suffix_family(sequences, token_counts, rng)
        cost = 0.5 + 0.005 * len(rewrite_map)
        return new_sequences, f"guided {suffix} -> {best_suffix} across {len(rewrite_map)} types", cost

    def _mutate_suffix_family_guided_sparse(
        self,
        sequences: list[list[str]],
        token_counts: Counter,
        guidance: BatchGuidance,
        rng: np.random.Generator,
    ) -> tuple[MutationPayload | None, str, float]:
        guided = self._guided_token_candidates(token_counts, guidance, min_len=4)
        target = self._sample_guided_token(guided, rng)
        if target is None:
            return self._mutate_suffix_family_sparse(sequences, token_counts, rng)

        suffix_len = int(rng.choice([2, 3, 4], p=[0.35, 0.45, 0.20]))
        suffix_len = min(suffix_len, len(target) - 1)
        suffix = target[-suffix_len:]
        family = [tok for tok in token_counts if tok.endswith(suffix) and len(tok) > suffix_len]
        if len(family) < 2:
            return self._mutate_suffix_family_sparse(sequences, token_counts, rng)

        base_score = sum(token_counts[tok] * self.latin_form_ref.score_token(tok) for tok in family)
        best_suffix = suffix
        best_score = base_score

        for _ in range(self.config.suffix_candidate_samples):
            candidate_suffix = self.latin_form_ref.sample_suffix(rng)
            if candidate_suffix == suffix:
                candidate_suffix = self._random_edit_token_form(suffix, rng)
            if candidate_suffix == suffix or len(candidate_suffix) < 1:
                continue
            candidate_score = sum(
                token_counts[tok] * self.latin_form_ref.score_token(tok[:-suffix_len] + candidate_suffix)
                for tok in family
            )
            if candidate_score > best_score:
                best_suffix = candidate_suffix
                best_score = candidate_score

        if best_suffix == suffix:
            return self._mutate_suffix_family_sparse(sequences, token_counts, rng)

        rewrite_map = {tok: tok[:-suffix_len] + best_suffix for tok in family}
        changed_sequences, affected = self._apply_token_rewrite_sparse(sequences, rewrite_map)
        if affected == 0:
            return self._mutate_suffix_family_sparse(sequences, token_counts, rng)
        cost = 0.5 + 0.005 * len(rewrite_map)
        return MutationPayload(changed_sequences=changed_sequences), f"guided {suffix} -> {best_suffix} across {len(rewrite_map)} types", cost

    def _mutate_split_token_guided(
        self,
        sequences: list[list[str]],
        token_counts: Counter,
        guidance: BatchGuidance,
        rng: np.random.Generator,
    ) -> tuple[list[list[str]] | None, str, float]:
        guided = self._guided_token_candidates(token_counts, guidance, min_len=6)
        tok = self._sample_guided_token(guided, rng)
        if tok is None:
            return super()._mutate_split_token(sequences, token_counts, rng)

        base_score = self.latin_form_ref.score_token(tok)
        best_parts: tuple[str, str] | None = None
        best_score = base_score

        for split_at in range(2, len(tok) - 1):
            left, right = tok[:split_at], tok[split_at:]
            if len(left) < 2 or len(right) < 2:
                continue
            candidate_score = self.latin_form_ref.score_token(left) + self.latin_form_ref.score_token(right)
            if candidate_score > best_score:
                best_parts = (left, right)
                best_score = candidate_score

        if best_parts is None:
            return super()._mutate_split_token(sequences, token_counts, rng)

        left, right = best_parts
        new_sequences = self._clone_sequences(sequences)
        replacements = 0
        for i, seq in enumerate(new_sequences):
            new_seq = []
            for item in seq:
                if item == tok:
                    new_seq.extend([left, right])
                    replacements += 1
                else:
                    new_seq.append(item)
            new_sequences[i] = new_seq

        if replacements == 0:
            return super()._mutate_split_token(sequences, token_counts, rng)
        return new_sequences, f"guided {tok} -> {left} + {right} ({replacements} occurrences)", 0.4

    def _mutate_split_token_guided_sparse(
        self,
        sequences: list[list[str]],
        token_counts: Counter,
        guidance: BatchGuidance,
        rng: np.random.Generator,
    ) -> tuple[MutationPayload | None, str, float]:
        guided = self._guided_token_candidates(token_counts, guidance, min_len=6)
        tok = self._sample_guided_token(guided, rng)
        if tok is None:
            return self._mutate_split_token_sparse(sequences, token_counts, rng)

        base_score = self.latin_form_ref.score_token(tok)
        best_parts: tuple[str, str] | None = None
        best_score = base_score

        for split_at in range(2, len(tok) - 1):
            left, right = tok[:split_at], tok[split_at:]
            if len(left) < 2 or len(right) < 2:
                continue
            candidate_score = self.latin_form_ref.score_token(left) + self.latin_form_ref.score_token(right)
            if candidate_score > best_score:
                best_parts = (left, right)
                best_score = candidate_score

        if best_parts is None:
            return self._mutate_split_token_sparse(sequences, token_counts, rng)

        left, right = best_parts
        changed_sequences, replacements = self._split_token_sparse(sequences, tok, left, right)
        if replacements == 0:
            return self._mutate_split_token_sparse(sequences, token_counts, rng)
        return MutationPayload(changed_sequences=changed_sequences), f"guided {tok} -> {left} + {right} ({replacements} occurrences)", 0.4

    def _mutate_function_word_burst_guided(
        self,
        sequences: list[list[str]],
        token_counts: Counter,
        guidance: BatchGuidance,
        rng: np.random.Generator,
    ) -> tuple[list[list[str]] | None, str, float]:
        candidates = self._guided_token_candidates(
            token_counts,
            guidance,
            min_len=2,
            max_len=4,
        )
        candidates = [(tok, count, weight) for tok, count, weight in candidates if count >= 3]
        if len(candidates) < self.config.function_burst_min_tokens:
            return super()._mutate_function_word_burst(sequences, token_counts, rng)

        weights = np.array(
            [count * max(0.2, 1.0 - self.latin_form_ref.score_token(tok)) * max(weight, 1e-6)
             for tok, count, weight in candidates],
            dtype=np.float64,
        )
        weights /= weights.sum()
        burst_size = int(
            rng.integers(
                self.config.function_burst_min_tokens,
                min(self.config.function_burst_max_tokens, len(candidates)) + 1,
            )
        )
        picked = rng.choice(len(candidates), size=burst_size, replace=False, p=weights)

        rewrite_map: dict[str, str] = {}
        details_parts: list[str] = []
        for raw_idx in np.atleast_1d(picked):
            tok = candidates[int(raw_idx)][0]
            new_tok = self._weirdify_token_form(tok, rng)
            if new_tok != tok:
                rewrite_map[tok] = new_tok
                details_parts.append(f"{tok}->{new_tok}")

        if not rewrite_map:
            return super()._mutate_function_word_burst(sequences, token_counts, rng)

        new_sequences, affected = self._apply_token_rewrite(sequences, rewrite_map)
        if affected == 0:
            return super()._mutate_function_word_burst(sequences, token_counts, rng)
        cost = 0.55 + 0.04 * len(rewrite_map)
        return new_sequences, f"guided {len(rewrite_map)} short tokens: {', '.join(details_parts[:5])}", cost

    def _mutate_function_word_burst_guided_sparse(
        self,
        sequences: list[list[str]],
        token_counts: Counter,
        guidance: BatchGuidance,
        rng: np.random.Generator,
    ) -> tuple[MutationPayload | None, str, float]:
        candidates = self._guided_token_candidates(
            token_counts,
            guidance,
            min_len=2,
            max_len=4,
        )
        candidates = [(tok, count, weight) for tok, count, weight in candidates if count >= 3]
        if len(candidates) < self.config.function_burst_min_tokens:
            return self._mutate_function_word_burst_sparse(sequences, token_counts, rng)

        weights = np.array(
            [count * max(0.2, 1.0 - self.latin_form_ref.score_token(tok)) * max(weight, 1e-6)
             for tok, count, weight in candidates],
            dtype=np.float64,
        )
        weights /= weights.sum()
        burst_size = int(
            rng.integers(
                self.config.function_burst_min_tokens,
                min(self.config.function_burst_max_tokens, len(candidates)) + 1,
            )
        )
        picked = rng.choice(len(candidates), size=burst_size, replace=False, p=weights)

        rewrite_map: dict[str, str] = {}
        details_parts: list[str] = []
        for raw_idx in np.atleast_1d(picked):
            tok = candidates[int(raw_idx)][0]
            new_tok = self._weirdify_token_form(tok, rng)
            if new_tok != tok:
                rewrite_map[tok] = new_tok
                details_parts.append(f"{tok}->{new_tok}")

        if not rewrite_map:
            return self._mutate_function_word_burst_sparse(sequences, token_counts, rng)

        changed_sequences, affected = self._apply_token_rewrite_sparse(sequences, rewrite_map)
        if affected == 0:
            return self._mutate_function_word_burst_sparse(sequences, token_counts, rng)
        cost = 0.55 + 0.04 * len(rewrite_map)
        return MutationPayload(changed_sequences=changed_sequences), f"guided {len(rewrite_map)} short tokens: {', '.join(details_parts[:5])}", cost

    def _mutate_paradigm_family_rewrite_guided(
        self,
        sequences: list[list[str]],
        token_counts: Counter,
        guidance: BatchGuidance,
        rng: np.random.Generator,
    ) -> tuple[list[list[str]] | None, str, float]:
        preferred_prefixes = {
            token[:prefix_len]
            for token in guidance.hotspot_tokens
            for prefix_len in range(
                self.config.paradigm_prefix_min_len,
                min(self.config.paradigm_prefix_max_len, max(len(token) - 1, 0)) + 1,
            )
            if len(token) > prefix_len + 1
        }
        if not preferred_prefixes:
            return super()._mutate_paradigm_family_rewrite(sequences, token_counts, rng)

        candidate_families: list[tuple[str, list[str]]] = []
        for prefix in sorted(preferred_prefixes):
            family = [
                tok for tok in token_counts
                if tok.startswith(prefix) and len(tok) > len(prefix) + 1
            ]
            family = sorted(set(family))
            if len(family) >= 2:
                candidate_families.append((prefix, family))
        if not candidate_families:
            return super()._mutate_paradigm_family_rewrite(sequences, token_counts, rng)

        weights = np.array(
            [sum(token_counts[tok] for tok in tokens) * len(tokens) for _, tokens in candidate_families],
            dtype=np.float64,
        )
        weights /= weights.sum()
        idx = int(rng.choice(len(candidate_families), p=weights))
        prefix, family = candidate_families[idx]

        sampled_suffixes = [
            self.latin_form_ref.sample_suffix(rng)
            for _ in range(self.config.suffix_candidate_samples)
        ]
        sampled_suffixes = [sfx for sfx in sampled_suffixes if len(sfx) >= 1]
        if not sampled_suffixes:
            return super()._mutate_paradigm_family_rewrite(sequences, token_counts, rng)

        rewrite_map: dict[str, str] = {}
        details_parts: list[str] = []
        for tok in family:
            suffix_len = min(max(2, len(tok) // 3), 4)
            stem = tok[:-suffix_len]
            best_tok = tok
            best_score = self.latin_form_ref.score_token(tok)
            for suffix in sampled_suffixes:
                candidate = stem + suffix
                score = self.latin_form_ref.score_token(candidate)
                if score > best_score and len(candidate) >= 2:
                    best_tok = candidate
                    best_score = score
            if best_tok != tok:
                rewrite_map[tok] = best_tok
                details_parts.append(f"{tok}->{best_tok}")

        if not rewrite_map:
            return super()._mutate_paradigm_family_rewrite(sequences, token_counts, rng)

        new_sequences, affected = self._apply_token_rewrite(sequences, rewrite_map)
        if affected == 0:
            return super()._mutate_paradigm_family_rewrite(sequences, token_counts, rng)
        cost = 0.75 + 0.02 * len(rewrite_map)
        return new_sequences, f"guided prefix {prefix} across {len(rewrite_map)} types", cost

    def _mutate_paradigm_family_rewrite_guided_sparse(
        self,
        sequences: list[list[str]],
        token_counts: Counter,
        guidance: BatchGuidance,
        rng: np.random.Generator,
    ) -> tuple[MutationPayload | None, str, float]:
        preferred_prefixes = {
            token[:prefix_len]
            for token in guidance.hotspot_tokens
            for prefix_len in range(
                self.config.paradigm_prefix_min_len,
                min(self.config.paradigm_prefix_max_len, max(len(token) - 1, 0)) + 1,
            )
            if len(token) > prefix_len + 1
        }
        if not preferred_prefixes:
            return self._mutate_paradigm_family_rewrite_sparse(sequences, token_counts, rng)

        candidate_families: list[tuple[str, list[str]]] = []
        for prefix in sorted(preferred_prefixes):
            family = [
                tok for tok in token_counts
                if tok.startswith(prefix) and len(tok) > len(prefix) + 1
            ]
            family = sorted(set(family))
            if len(family) >= 2:
                candidate_families.append((prefix, family))
        if not candidate_families:
            return self._mutate_paradigm_family_rewrite_sparse(sequences, token_counts, rng)

        weights = np.array(
            [sum(token_counts[tok] for tok in tokens) * len(tokens) for _, tokens in candidate_families],
            dtype=np.float64,
        )
        weights /= weights.sum()
        idx = int(rng.choice(len(candidate_families), p=weights))
        prefix, family = candidate_families[idx]

        sampled_suffixes = [
            self.latin_form_ref.sample_suffix(rng)
            for _ in range(self.config.suffix_candidate_samples)
        ]
        sampled_suffixes = [sfx for sfx in sampled_suffixes if len(sfx) >= 1]
        if not sampled_suffixes:
            return self._mutate_paradigm_family_rewrite_sparse(sequences, token_counts, rng)

        rewrite_map: dict[str, str] = {}
        details_parts: list[str] = []
        for tok in family:
            suffix_len = min(max(2, len(tok) // 3), 4)
            stem = tok[:-suffix_len]
            best_tok = tok
            best_score = self.latin_form_ref.score_token(tok)
            for suffix in sampled_suffixes:
                candidate = stem + suffix
                score = self.latin_form_ref.score_token(candidate)
                if score > best_score and len(candidate) >= 2:
                    best_tok = candidate
                    best_score = score
            if best_tok != tok:
                rewrite_map[tok] = best_tok
                details_parts.append(f"{tok}->{best_tok}")

        if not rewrite_map:
            return self._mutate_paradigm_family_rewrite_sparse(sequences, token_counts, rng)

        changed_sequences, affected = self._apply_token_rewrite_sparse(sequences, rewrite_map)
        if affected == 0:
            return self._mutate_paradigm_family_rewrite_sparse(sequences, token_counts, rng)
        cost = 0.75 + 0.02 * len(rewrite_map)
        return MutationPayload(changed_sequences=changed_sequences), f"guided prefix {prefix} across {len(rewrite_map)} types", cost

    def _mutate_sequence_span_rewrite_guided(
        self,
        sequences: list[list[str]],
        guidance: BatchGuidance,
        rng: np.random.Generator,
    ) -> tuple[list[list[str]] | None, str, float]:
        hotspot_set = set(guidance.hotspot_tokens)
        if not hotspot_set:
            return super()._mutate_sequence_span_rewrite(sequences, rng)

        min_span = max(1, self.config.span_min_sequences)
        max_span = min(self.config.span_max_sequences, len(sequences))
        if len(sequences) < min_span or max_span < min_span:
            return super()._mutate_sequence_span_rewrite(sequences, rng)

        hotspot_indices = [i for i, seq in enumerate(sequences) if any(tok in hotspot_set for tok in seq)]
        if not hotspot_indices:
            return super()._mutate_sequence_span_rewrite(sequences, rng)

        span_len = int(rng.integers(min_span, max_span + 1))
        center = int(rng.choice(hotspot_indices))
        start_min = max(0, center - span_len + 1)
        start_max = min(center, len(sequences) - span_len)
        start = start_min if start_min == start_max else int(rng.integers(start_min, start_max + 1))

        span = self._clone_sequences(sequences[start : start + span_len])
        mutated_span, details, cost = super()._mutate_sequence_span_rewrite(span, rng)
        if mutated_span is None or mutated_span == span:
            return super()._mutate_sequence_span_rewrite(sequences, rng)

        new_sequences = self._splice_sequence_span(sequences, start, mutated_span)
        return new_sequences, f"guided span[{start}:{start + span_len}] {details}", cost

    def _mutate_sequence_span_rewrite_sparse(
        self,
        sequences: list[list[str]],
        rng: np.random.Generator,
    ) -> tuple[MutationPayload | None, str, float]:
        min_span = max(1, self.config.span_min_sequences)
        max_span = min(self.config.span_max_sequences, len(sequences))
        if len(sequences) < min_span or max_span < min_span:
            return None, "not enough sequences for span rewrite", 0.0

        span_len = int(rng.integers(min_span, max_span + 1))
        start = int(rng.integers(0, len(sequences) - span_len + 1))
        span = self._clone_sequences(sequences[start : start + span_len])
        mutated_span, detail_text, accumulated_cost = self._rewrite_sequence_span_local(span, rng)
        if mutated_span is None:
            return None, f"span[{start}:{start + span_len}] unchanged", 0.0

        changed_sequences = self._splice_sequence_span_sparse(sequences, start, mutated_span)
        if not changed_sequences:
            return None, f"span[{start}:{start + span_len}] unchanged", 0.0

        total_cost = 0.75 + accumulated_cost + 0.08 * max(span_len - 1, 0)
        return MutationPayload(changed_sequences=changed_sequences), f"span[{start}:{start + span_len}] {detail_text}", total_cost

    def _mutate_sequence_span_rewrite_guided_sparse(
        self,
        sequences: list[list[str]],
        guidance: BatchGuidance,
        rng: np.random.Generator,
    ) -> tuple[MutationPayload | None, str, float]:
        hotspot_set = set(guidance.hotspot_tokens)
        if not hotspot_set:
            return self._mutate_sequence_span_rewrite_sparse(sequences, rng)

        min_span = max(1, self.config.span_min_sequences)
        max_span = min(self.config.span_max_sequences, len(sequences))
        if len(sequences) < min_span or max_span < min_span:
            return self._mutate_sequence_span_rewrite_sparse(sequences, rng)

        hotspot_indices = [i for i, seq in enumerate(sequences) if any(tok in hotspot_set for tok in seq)]
        if not hotspot_indices:
            return self._mutate_sequence_span_rewrite_sparse(sequences, rng)

        span_len = int(rng.integers(min_span, max_span + 1))
        center = int(rng.choice(hotspot_indices))
        start_min = max(0, center - span_len + 1)
        start_max = min(center, len(sequences) - span_len)
        start = start_min if start_min == start_max else int(rng.integers(start_min, start_max + 1))

        span = self._clone_sequences(sequences[start : start + span_len])
        mutated_span, detail_text, accumulated_cost = self._rewrite_sequence_span_local(span, rng)
        if mutated_span is None:
            return self._mutate_sequence_span_rewrite_sparse(sequences, rng)

        changed_sequences = self._splice_sequence_span_sparse(sequences, start, mutated_span)
        if not changed_sequences:
            return self._mutate_sequence_span_rewrite_sparse(sequences, rng)

        total_cost = 0.75 + accumulated_cost + 0.08 * max(span_len - 1, 0)
        return MutationPayload(changed_sequences=changed_sequences), f"guided span[{start}:{start + span_len}] {detail_text}", total_cost

    def _apply_named_operator_guided(
        self,
        operator: str,
        sequences: list[list[str]],
        token_counts: Counter,
        bigram_counts: Counter,
        guidance: BatchGuidance,
        rng: np.random.Generator,
    ) -> tuple[list[list[str]] | None, str, float]:
        if operator == "token_char_edit":
            return self._mutate_token_char_edit_guided(sequences, token_counts, guidance, rng)
        if operator == "suffix_family_rewrite":
            return self._mutate_suffix_family_guided(sequences, token_counts, guidance, rng)
        if operator == "split_token":
            return self._mutate_split_token_guided(sequences, token_counts, guidance, rng)
        if operator == "sequence_span_rewrite":
            return self._mutate_sequence_span_rewrite_guided(sequences, guidance, rng)
        if operator == "function_word_burst":
            return self._mutate_function_word_burst_guided(sequences, token_counts, guidance, rng)
        if operator == "paradigm_family_rewrite":
            return self._mutate_paradigm_family_rewrite_guided(sequences, token_counts, guidance, rng)
        return super()._apply_named_operator(operator, sequences, token_counts, bigram_counts, rng)

    def _apply_named_operator_guided_payload(
        self,
        operator: str,
        sequences: list[list[str]],
        token_counts: Counter,
        bigram_counts: Counter,
        guidance: BatchGuidance,
        rng: np.random.Generator,
    ) -> tuple[MutationPayload | None, str, float]:
        if operator == "token_char_edit":
            return self._mutate_token_char_edit_guided_sparse(sequences, token_counts, guidance, rng)
        if operator == "suffix_family_rewrite":
            return self._mutate_suffix_family_guided_sparse(sequences, token_counts, guidance, rng)
        if operator == "split_token":
            return self._mutate_split_token_guided_sparse(sequences, token_counts, guidance, rng)
        if operator == "sequence_span_rewrite":
            return self._mutate_sequence_span_rewrite_guided_sparse(sequences, guidance, rng)
        if operator == "function_word_burst":
            return self._mutate_function_word_burst_guided_sparse(sequences, token_counts, guidance, rng)
        if operator == "paradigm_family_rewrite":
            return self._mutate_paradigm_family_rewrite_guided_sparse(sequences, token_counts, guidance, rng)
        mutated, details, cost = super()._apply_named_operator(operator, sequences, token_counts, bigram_counts, rng)
        payload = None if mutated is None else MutationPayload(sequences=mutated)
        return payload, details, cost

    def _apply_named_operator_payload(
        self,
        operator: str,
        sequences: list[list[str]],
        token_counts: Counter,
        bigram_counts: Counter,
        rng: np.random.Generator,
    ) -> tuple[MutationPayload | None, str, float]:
        if operator == "token_char_edit":
            return self._mutate_token_char_edit_sparse(sequences, token_counts, rng)
        if operator == "suffix_family_rewrite":
            return self._mutate_suffix_family_sparse(sequences, token_counts, rng)
        if operator == "split_token":
            return self._mutate_split_token_sparse(sequences, token_counts, rng)
        if operator == "sequence_span_rewrite":
            return self._mutate_sequence_span_rewrite_sparse(sequences, rng)
        if operator == "function_word_burst":
            return self._mutate_function_word_burst_sparse(sequences, token_counts, rng)
        if operator == "paradigm_family_rewrite":
            return self._mutate_paradigm_family_rewrite_sparse(sequences, token_counts, rng)
        mutated, details, cost = self._apply_named_operator(operator, sequences, token_counts, bigram_counts, rng)
        payload = None if mutated is None else MutationPayload(sequences=mutated)
        return payload, details, cost

    def _apply_mutation_payload_in_place(
        self,
        *,
        base_sequences: list[list[str]],
        working_sequences: list[list[str]],
        changed_indices: set[int],
        payload: MutationPayload,
    ) -> bool:
        """
        Apply a candidate payload into a shallow working corpus view.

        This is the key v5.1 macro-bundle optimization: bundle sub-steps can
        stay sparse while we keep one mutable top-level sequence view for the
        current bundle candidate.
        """
        applied = False
        if payload.changed_sequences is not None:
            for idx, seq in payload.changed_sequences.items():
                next_seq = list(seq)
                if working_sequences[idx] != next_seq:
                    working_sequences[idx] = next_seq
                    applied = True
                if base_sequences[idx] != working_sequences[idx]:
                    changed_indices.add(idx)
                else:
                    changed_indices.discard(idx)
            return applied

        if payload.sequences is not None:
            for idx, seq in enumerate(payload.sequences):
                next_seq = list(seq)
                if working_sequences[idx] != next_seq:
                    working_sequences[idx] = next_seq
                    applied = True
                if base_sequences[idx] != working_sequences[idx]:
                    changed_indices.add(idx)
                else:
                    changed_indices.discard(idx)
        return applied

    def _mutate_macro_bundle_rewrite_payload(
        self,
        sequences: list[list[str]],
        rng: np.random.Generator,
        *,
        guidance: BatchGuidance | None = None,
    ) -> tuple[MutationPayload | None, str, float]:
        sub_ops = (
            "token_char_edit",
            "suffix_family_rewrite",
            "sequence_span_rewrite",
            "function_word_burst",
            "paradigm_family_rewrite",
            "split_token",
        )
        weights = np.array([0.16, 0.16, 0.20, 0.18, 0.18, 0.12], dtype=np.float64)
        weights /= weights.sum()

        working_sequences = list(sequences)
        changed_indices: set[int] = set()
        details_parts: list[str] = []
        total_cost = 0.6

        n_steps = int(
            rng.integers(
                self.config.macro_bundle_min_steps,
                self.config.macro_bundle_max_steps + 1,
            )
        )
        for _ in range(n_steps):
            token_counts = self._token_counts(working_sequences)
            bigram_counts = self._bigram_counts(working_sequences)
            operator = str(rng.choice(sub_ops, p=weights))
            if guidance is None:
                payload, details, cost = self._apply_named_operator_payload(
                    operator,
                    working_sequences,
                    token_counts,
                    bigram_counts,
                    rng,
                )
            else:
                payload, details, cost = self._apply_named_operator_guided_payload(
                    operator,
                    working_sequences,
                    token_counts,
                    bigram_counts,
                    guidance,
                    rng,
                )
            if payload is None:
                continue
            if not self._apply_mutation_payload_in_place(
                base_sequences=sequences,
                working_sequences=working_sequences,
                changed_indices=changed_indices,
                payload=payload,
            ):
                continue
            total_cost += cost
            details_parts.append(f"{operator}:{details}")

        if not changed_indices:
            return None, "macro bundle produced no change", 0.0

        changed_sequences = {
            idx: list(working_sequences[idx])
            for idx in sorted(changed_indices)
            if sequences[idx] != working_sequences[idx]
        }
        if not changed_sequences:
            return None, "macro bundle produced no change", 0.0
        return MutationPayload(changed_sequences=changed_sequences), "; ".join(details_parts[:5]), total_cost

    def _mutate_macro_bundle_rewrite_guided(
        self,
        sequences: list[list[str]],
        guidance: BatchGuidance,
        rng: np.random.Generator,
    ) -> tuple[list[list[str]] | None, str, float]:
        payload, details, cost = self._mutate_macro_bundle_rewrite_payload(
            sequences,
            rng,
            guidance=guidance,
        )
        if payload is None:
            return self._mutate_macro_bundle_rewrite(sequences, rng)
        return payload.materialize(sequences), details, cost

    def _mutate_macro_bundle_rewrite(
        self,
        sequences: list[list[str]],
        rng: np.random.Generator,
    ) -> tuple[list[list[str]] | None, str, float]:
        payload, details, cost = self._mutate_macro_bundle_rewrite_payload(sequences, rng)
        if payload is None:
            return None, details, cost
        return payload.materialize(sequences), details, cost

    def _mutate_macro_bundle_rewrite_sparse(
        self,
        sequences: list[list[str]],
        rng: np.random.Generator,
    ) -> tuple[MutationPayload | None, str, float]:
        return self._mutate_macro_bundle_rewrite_payload(sequences, rng)

    def _mutate_macro_bundle_rewrite_guided_sparse(
        self,
        sequences: list[list[str]],
        guidance: BatchGuidance,
        rng: np.random.Generator,
    ) -> tuple[MutationPayload | None, str, float]:
        payload, details, cost = self._mutate_macro_bundle_rewrite_payload(
            sequences,
            rng,
            guidance=guidance,
        )
        if payload is None:
            return self._mutate_macro_bundle_rewrite_sparse(sequences, rng)
        return payload, details, cost

    def _mutate_token_char_edit_sparse(
        self,
        sequences: list[list[str]],
        token_counts: Counter,
        rng: np.random.Generator,
    ) -> tuple[MutationPayload | None, str, float]:
        candidates = [(tok, count) for tok, count in token_counts.items() if len(tok) >= 3]
        if not candidates:
            return None, "no eligible token", 0.0

        weights = np.array([count * max(count, 1) * max(len(tok), 1) for tok, count in candidates], dtype=np.float64)
        weights /= weights.sum()
        idx = int(rng.choice(len(candidates), p=weights))
        tok = candidates[idx][0]
        new_tok = self._edit_token_form(tok, rng)
        if new_tok == tok:
            return None, f"token {tok} unchanged", 0.0

        changed_sequences, affected = self._apply_token_rewrite_sparse(sequences, {tok: new_tok})
        if affected == 0:
            return None, f"token {tok} had no occurrences", 0.0
        return MutationPayload(changed_sequences=changed_sequences), f"{tok} -> {new_tok} ({affected} occurrences)", 0.25

    def _mutate_suffix_family_sparse(
        self,
        sequences: list[list[str]],
        token_counts: Counter,
        rng: np.random.Generator,
    ) -> tuple[MutationPayload | None, str, float]:
        tokens = [tok for tok in token_counts if len(tok) >= 4]
        if not tokens:
            return None, "no suffix-family candidates", 0.0

        target = str(rng.choice(tokens))
        suffix_len = int(rng.choice([2, 3, 4], p=[0.35, 0.45, 0.20]))
        suffix_len = min(suffix_len, len(target) - 1)
        suffix = target[-suffix_len:]
        family = [tok for tok in token_counts if tok.endswith(suffix) and len(tok) > suffix_len]
        if len(family) < 2:
            return None, f"suffix family too small for {suffix}", 0.0

        base_score = sum(
            token_counts[tok] * self.latin_form_ref.score_token(tok)
            for tok in family
        )
        best_suffix = suffix
        best_score = base_score

        for _ in range(self.config.suffix_candidate_samples):
            candidate_suffix = self.latin_form_ref.sample_suffix(rng)
            if candidate_suffix == suffix:
                candidate_suffix = self._random_edit_token_form(suffix, rng)
            if candidate_suffix == suffix or len(candidate_suffix) < 1:
                continue
            candidate_score = sum(
                token_counts[tok] * self.latin_form_ref.score_token(tok[:-suffix_len] + candidate_suffix)
                for tok in family
            )
            if candidate_score > best_score:
                best_suffix = candidate_suffix
                best_score = candidate_score

        if best_suffix == suffix:
            return None, f"suffix {suffix} unchanged", 0.0

        rewrite_map = {tok: tok[:-suffix_len] + best_suffix for tok in family}
        changed_sequences, affected = self._apply_token_rewrite_sparse(sequences, rewrite_map)
        if affected == 0:
            return None, f"suffix family {suffix} had no affected tokens", 0.0
        cost = 0.5 + 0.005 * len(rewrite_map)
        details = f"{suffix} -> {best_suffix} across {len(rewrite_map)} token types, {affected} occurrences"
        return MutationPayload(changed_sequences=changed_sequences), details, cost

    def _mutate_split_token_sparse(
        self,
        sequences: list[list[str]],
        token_counts: Counter,
        rng: np.random.Generator,
    ) -> tuple[MutationPayload | None, str, float]:
        tokens = [(tok, count) for tok, count in token_counts.items() if len(tok) >= 6]
        if not tokens:
            return None, "no splittable tokens", 0.0

        weights = np.array([count * len(tok) for tok, count in tokens], dtype=np.float64)
        weights /= weights.sum()
        idx = int(rng.choice(len(tokens), p=weights))
        tok = tokens[idx][0]
        base_score = self.latin_form_ref.score_token(tok)
        best_parts: tuple[str, str] | None = None
        best_score = base_score

        for split_at in range(2, len(tok) - 1):
            left, right = tok[:split_at], tok[split_at:]
            if len(left) < 2 or len(right) < 2:
                continue
            candidate_score = (
                self.latin_form_ref.score_token(left)
                + self.latin_form_ref.score_token(right)
            )
            if candidate_score > best_score:
                best_parts = (left, right)
                best_score = candidate_score

        if best_parts is None:
            return None, f"no beneficial split for {tok}", 0.0

        left, right = best_parts
        changed_sequences, replacements = self._split_token_sparse(sequences, tok, left, right)
        if replacements == 0:
            return None, f"token {tok} not split", 0.0
        return MutationPayload(changed_sequences=changed_sequences), f"{tok} -> {left} + {right} ({replacements} occurrences)", 0.4

    def _mutate_function_word_burst_sparse(
        self,
        sequences: list[list[str]],
        token_counts: Counter,
        rng: np.random.Generator,
    ) -> tuple[MutationPayload | None, str, float]:
        candidates = [
            (tok, count)
            for tok, count in token_counts.items()
            if 2 <= len(tok) <= 4 and count >= 3
        ]
        if len(candidates) < self.config.function_burst_min_tokens:
            return None, "not enough function-word burst candidates", 0.0

        weights = np.array(
            [count * max(0.2, 1.0 - self.latin_form_ref.score_token(tok)) for tok, count in candidates],
            dtype=np.float64,
        )
        weights /= weights.sum()
        burst_size = int(
            rng.integers(
                self.config.function_burst_min_tokens,
                min(self.config.function_burst_max_tokens, len(candidates)) + 1,
            )
        )
        picked = rng.choice(len(candidates), size=burst_size, replace=False, p=weights)

        rewrite_map: dict[str, str] = {}
        details_parts: list[str] = []
        for raw_idx in np.atleast_1d(picked):
            tok = candidates[int(raw_idx)][0]
            new_tok = self._weirdify_token_form(tok, rng)
            if new_tok != tok:
                rewrite_map[tok] = new_tok
                details_parts.append(f"{tok}->{new_tok}")

        if not rewrite_map:
            return None, "function-word burst produced no rewrites", 0.0

        changed_sequences, affected = self._apply_token_rewrite_sparse(sequences, rewrite_map)
        if affected == 0:
            return None, "function-word burst affected nothing", 0.0
        cost = 0.55 + 0.04 * len(rewrite_map)
        return MutationPayload(changed_sequences=changed_sequences), f"{len(rewrite_map)} short tokens: {', '.join(details_parts[:5])}", cost

    def _mutate_paradigm_family_rewrite_sparse(
        self,
        sequences: list[list[str]],
        token_counts: Counter,
        rng: np.random.Generator,
    ) -> tuple[MutationPayload | None, str, float]:
        families: dict[str, list[str]] = {}
        for tok in token_counts:
            if len(tok) < 5:
                continue
            for prefix_len in range(self.config.paradigm_prefix_min_len, self.config.paradigm_prefix_max_len + 1):
                if len(tok) > prefix_len + 1:
                    families.setdefault(tok[:prefix_len], []).append(tok)

        candidate_families = [
            (prefix, sorted(set(tokens)))
            for prefix, tokens in families.items()
            if len(set(tokens)) >= 2
        ]
        if not candidate_families:
            return None, "no paradigm-family candidates", 0.0

        weights = np.array(
            [sum(token_counts[tok] for tok in tokens) * len(tokens) for _, tokens in candidate_families],
            dtype=np.float64,
        )
        weights /= weights.sum()
        idx = int(rng.choice(len(candidate_families), p=weights))
        prefix, family = candidate_families[idx]

        sampled_suffixes = [
            self.latin_form_ref.sample_suffix(rng)
            for _ in range(self.config.suffix_candidate_samples)
        ]
        sampled_suffixes = [sfx for sfx in sampled_suffixes if len(sfx) >= 1]
        if not sampled_suffixes:
            return None, f"no sampled suffixes for family {prefix}", 0.0

        rewrite_map: dict[str, str] = {}
        details_parts: list[str] = []
        for tok in family:
            suffix_len = min(max(2, len(tok) // 3), 4)
            stem = tok[:-suffix_len]
            best_tok = tok
            best_score = self.latin_form_ref.score_token(tok)
            for suffix in sampled_suffixes:
                candidate = stem + suffix
                score = self.latin_form_ref.score_token(candidate)
                if score > best_score and len(candidate) >= 2:
                    best_tok = candidate
                    best_score = score
            if best_tok != tok:
                rewrite_map[tok] = best_tok
                details_parts.append(f"{tok}->{best_tok}")

        if not rewrite_map:
            return None, f"family {prefix} produced no rewrites", 0.0

        changed_sequences, affected = self._apply_token_rewrite_sparse(sequences, rewrite_map)
        if affected == 0:
            return None, f"family {prefix} affected nothing", 0.0
        cost = 0.75 + 0.02 * len(rewrite_map)
        return MutationPayload(changed_sequences=changed_sequences), f"prefix {prefix} across {len(rewrite_map)} types: {', '.join(details_parts[:4])}", cost

    def _mutate_candidate(
        self,
        sequences: list[list[str]],
        rng: np.random.Generator,
        guidance: BatchGuidance | None = None,
        precomputed_token_counts: Counter | None = None,
        precomputed_bigram_counts: Counter | None = None,
    ) -> tuple[MutationPayload | None, str, str, float]:
        token_counts = precomputed_token_counts if precomputed_token_counts is not None else self._token_counts(sequences)
        bigram_counts = precomputed_bigram_counts if precomputed_bigram_counts is not None else self._bigram_counts(sequences)

        for _ in range(10):
            payload: MutationPayload | None = None
            operator = self._choose_operator(rng)
            if guidance is not None and operator == "macro_bundle_rewrite":
                payload, details, cost = self._mutate_macro_bundle_rewrite_guided_sparse(sequences, guidance, rng)
            elif guidance is not None:
                payload, details, cost = self._apply_named_operator_guided_payload(
                    operator,
                    sequences,
                    token_counts,
                    bigram_counts,
                    guidance,
                    rng,
                )
            elif operator == "macro_bundle_rewrite":
                payload, details, cost = self._mutate_macro_bundle_rewrite_sparse(sequences, rng)
            else:
                payload, details, cost = self._apply_named_operator_payload(
                    operator,
                    sequences,
                    token_counts,
                    bigram_counts,
                    rng,
                )

            if payload is not None:
                return payload, operator, details, cost

        return None, "none", "no valid mutation generated", 0.0

    def _candidate_state_from_scores(
        self,
        sequences: list[list[str]],
        scores: CandidateScores,
        mutation_cost: float,
    ) -> CandidateState:
        """Build a CandidateState from IncrementalScoringState.evaluate() output."""
        return CandidateState(
            sequences=sequences,
            operator="",
            details="",
            mutation_cost=mutation_cost,
            structural_vector=scores.structural_vector,
            latin_structural_score=scores.latin_structural_score,
            latin_form_score=scores.latin_form_score,
            form_details=scores.form_details,
            total_score=scores.total_score,
            scores=scores.scores,
            diagnostics=scores.diagnostics,
            type_token_ratio=scores.type_token_ratio,
            bigram_coverage=scores.bigram_coverage,
            trigram_coverage=scores.trigram_coverage,
            bigram_profile=scores.bigram_profile,
            trigram_profile=scores.trigram_profile,
        )

    def _reset_live_events(self) -> None:
        """Start a fresh rolling event stream for this block run."""
        self._live_event_buffer = []
        try:
            self._live_events_path.unlink(missing_ok=True)
        except Exception:
            log.debug("Could not clear live event stream at %s", self._live_events_path, exc_info=True)

    def _flush_live_events(self) -> None:
        """Flush buffered live events to disk in one append operation."""
        if not self._live_event_buffer:
            return
        self._live_events_path.parent.mkdir(parents=True, exist_ok=True)
        payload = "".join(json.dumps(event, ensure_ascii=False) + "\n" for event in self._live_event_buffer)
        with self._live_events_path.open("a", encoding="utf-8") as fh:
            fh.write(payload)
        self._live_event_buffer.clear()

    def _should_emit_live_event(self, outcome: str) -> bool:
        mode = getattr(self.config, "live_event_mode", "all")
        if mode == "off":
            return False
        if mode == "selected":
            return outcome in {"accepted", "best_rejected", "stage_committed", "no_valid_mutation"}
        if mode == "accepted_only":
            return outcome in {"accepted", "stage_committed"}
        return True

    def _emit_live_event(
        self,
        *,
        proposal_index: int,
        candidate_index: int,
        outcome: str,
        operator: str,
        details: str,
        mutation_cost: float,
        candidate: CandidateState | None = None,
        score_delta: float | None = None,
        stage_id: str | None = None,
        backend: str | None = None,
    ) -> None:
        """Append one human-debuggable candidate event for the live TUI."""
        if not self._should_emit_live_event(outcome):
            return
        event = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "proposal_index": int(proposal_index),
            "candidate_index": int(candidate_index),
            "outcome": outcome,
            "operator": operator,
            "details": details,
            "mutation_cost": float(mutation_cost),
            "stage_id": stage_id,
            "backend": backend,
        }
        if score_delta is not None:
            event["score_delta"] = float(score_delta)
        if candidate is not None:
            event.update(
                {
                    "total_score": float(candidate.total_score),
                    "latin_structural_score": float(candidate.latin_structural_score),
                    "latin_form_score": float(candidate.latin_form_score),
                    "family_alignment_score": float(candidate.diagnostics.get("family_alignment_score", 0.0)),
                }
            )
        self._live_event_buffer.append(event)
        if len(self._live_event_buffer) >= max(1, int(getattr(self.config, "live_event_buffer_size", 64))):
            self._flush_live_events()

    def _batch_scoring_backend(self) -> str:
        if not self.config.use_fortran_batch:
            return "python"
        if self._fortran_cosine_scorer is None:
            return "python_fallback"
        return "fortran_batch" if self._fortran_cosine_scorer.using_fortran else "numpy_batch"

    def _score_mutation_pool(
        self,
        *,
        current: CandidateState,
        mutation_pool: list[tuple[MutationPayload, str, str, float]],
        proposal_guidance: BatchGuidance | None,
    ) -> list[tuple[CandidateState, str, str, float, MutationPayload]]:
        """
        Score all generated candidates for one proposal.

        If batch scoring is enabled and an incremental state exists, batch the
        form-score computation across the proposal's candidate set using the
        Fortran/numpy cosine scorer. Otherwise fall back to the reference path.
        """
        cfg = self.config
        scored: list[tuple[CandidateState, str, str, float, MutationPayload]] = []
        use_batch = bool(cfg.use_fortran_batch and self._scoring_state is not None and self._fortran_cosine_scorer is not None)
        backend = self._batch_scoring_backend()

        if use_batch:
            sparse_pool = [
                (payload, operator, details, mutation_cost)
                for payload, operator, details, mutation_cost in mutation_pool
                if payload.is_sparse
            ]
            full_pool = [
                (payload, operator, details, mutation_cost)
                for payload, operator, details, mutation_cost in mutation_pool
                if not payload.is_sparse
            ]

            if sparse_pool:
                batch_scores = self._scoring_state.evaluate_batch_changed_sequences(
                    [payload.changed_sequences or {} for payload, _, _, _ in sparse_pool],
                    [mutation_cost for _, _, _, mutation_cost in sparse_pool],
                    cfg.form_weight,
                    cfg.coherence_weight,
                    cfg.mutation_cost_weight,
                )
                for (payload, operator, details, mutation_cost), scores in zip(sparse_pool, batch_scores):
                    candidate = self._candidate_state_from_scores(current.sequences, scores, mutation_cost)
                    if self._transparency_scorer is not None and cfg.transparency_weight > 0.0:
                        materialized = payload.materialize(current.sequences)
                        t_score = self._transparency_scorer.score(materialized)
                        candidate.total_score += cfg.transparency_weight * t_score
                        candidate.diagnostics["transparency_score"] = float(t_score)
                    candidate = self._amplify_reward(current, candidate)
                    candidate.operator = operator
                    candidate.details = details
                    candidate.diagnostics = {
                        **candidate.diagnostics,
                        "candidate_scoring_backend": backend,
                        "candidate_payload_mode": "sparse",
                    }
                    if proposal_guidance is not None:
                        candidate.diagnostics = {
                            **candidate.diagnostics,
                            **proposal_guidance.diagnostics(),
                        }
                    scored.append((candidate, operator, details, mutation_cost, payload))

            if full_pool:
                batch_scores = self._scoring_state.evaluate_batch(
                    [payload.sequences for payload, _, _, _ in full_pool if payload.sequences is not None],
                    [mutation_cost for _, _, _, mutation_cost in full_pool],
                    cfg.form_weight,
                    cfg.coherence_weight,
                    cfg.mutation_cost_weight,
                )
                for (payload, operator, details, mutation_cost), scores in zip(full_pool, batch_scores):
                    materialized = payload.sequences or current.sequences
                    candidate = self._candidate_state_from_scores(materialized, scores, mutation_cost)
                    if self._transparency_scorer is not None and cfg.transparency_weight > 0.0:
                        t_score = self._transparency_scorer.score(materialized)
                        candidate.total_score += cfg.transparency_weight * t_score
                        candidate.diagnostics["transparency_score"] = float(t_score)
                    candidate = self._amplify_reward(current, candidate)
                    candidate.operator = operator
                    candidate.details = details
                    candidate.diagnostics = {
                        **candidate.diagnostics,
                        "candidate_scoring_backend": backend,
                        "candidate_payload_mode": "materialized",
                    }
                    if proposal_guidance is not None:
                        candidate.diagnostics = {
                            **candidate.diagnostics,
                            **proposal_guidance.diagnostics(),
                        }
                    scored.append((candidate, operator, details, mutation_cost, payload))
            return scored

        for payload, operator, details, mutation_cost in mutation_pool:
            materialized = payload.sequences
            if self._scoring_state is not None and payload.is_sparse:
                scores = self._scoring_state.evaluate_changed_sequences(
                    payload.changed_sequences or {},
                    mutation_cost,
                    cfg.form_weight,
                    cfg.coherence_weight,
                    cfg.mutation_cost_weight,
                )
                candidate = self._candidate_state_from_scores(current.sequences, scores, mutation_cost)
                payload_mode = "sparse"
            elif self._scoring_state is not None:
                scores = self._scoring_state.evaluate(
                    materialized,
                    mutation_cost,
                    cfg.form_weight,
                    cfg.coherence_weight,
                    cfg.mutation_cost_weight,
                )
                candidate = self._candidate_state_from_scores(materialized, scores, mutation_cost)
                payload_mode = "materialized"
            else:
                materialized = payload.materialize(current.sequences)
                candidate = self._evaluate_sequences(materialized, mutation_cost=mutation_cost)
                payload_mode = "materialized"

            if self._transparency_scorer is not None and cfg.transparency_weight > 0.0:
                if materialized is None:
                    materialized = payload.materialize(current.sequences)
                t_score = self._transparency_scorer.score(materialized)
                candidate.total_score += cfg.transparency_weight * t_score
                candidate.diagnostics["transparency_score"] = float(t_score)

            candidate = self._amplify_reward(current, candidate)
            candidate.operator = operator
            candidate.details = details
            candidate.diagnostics = {
                **candidate.diagnostics,
                "candidate_scoring_backend": backend,
                "candidate_payload_mode": payload_mode,
            }
            if proposal_guidance is not None:
                candidate.diagnostics = {
                    **candidate.diagnostics,
                    **proposal_guidance.diagnostics(),
                }
            scored.append((candidate, operator, details, mutation_cost, payload))
        return scored

    def _save_stage(self, candidate, stage_id, iteration, proposal_index,
                    parent_stage_id, mutation_operator, mutation_details,
                    save_dense_matrices: bool | None = None) -> ReinforcedV2StageRecord:
        if save_dense_matrices is None:
            save_dense_matrices = self.config.save_dense_matrices
        return super()._save_stage(
            candidate, stage_id, iteration, proposal_index,
            parent_stage_id, mutation_operator, mutation_details,
            save_dense_matrices=save_dense_matrices,
        )

    def run(self) -> list[ReinforcedV2StageRecord]:
        cfg = self.config
        rng = np.random.default_rng(cfg.seed)
        self._reset_live_events()

        log.info(
            "Relational reinforced v4: language=%s, num_sequences=%d, proposals=%d, candidates=%d",
            self.language, cfg.num_sequences, cfg.max_proposals, cfg.n_candidates,
        )

        current_sequences = self._sample_initial_corpus(rng)
        current = self._evaluate_sequences(current_sequences, mutation_cost=0.0)
        current.diagnostics = {
            **current.diagnostics,
            "reward_struct_gain": 0.0,
            "reward_form_gain": 0.0,
            "reward_suffix_gain": 0.0,
            "reward_trigram_gain": 0.0,
            "reward_coherence_gain": 0.0,
            "reward_bonus": 0.0,
            "reward_penalty_relief": 0.0,
            "reward_effective_total_score": current.total_score,
        }
        current_tensor_state = self._initialize_batch_guidance_state(current.sequences)
        if current_tensor_state is not None:
            current.diagnostics = {
                **current.diagnostics,
                "batch_guidance_tensor_state_update_mode": "seed_build",
                "batch_guidance_tensor_anchor_vocab_size": len(current_tensor_state.idx2token),
            }

        # --- Incremental scoring state ---
        if cfg.use_incremental_scoring:
            self._scoring_state = IncrementalScoringState.from_sequences(
                current_sequences,
                self.latin_form_ref,
                self.latin_structural_ref,
                self._references,
                fortran_cosine_scorer=self._fortran_cosine_scorer,
            )
            log.info(
                "IncrementalScoringState initialized (use_incremental_scoring=True, fortran_cosine=%s)",
                self._fortran_cosine_scorer is not None,
            )
        else:
            self._scoring_state = None

        self._annotate_alignment(current)

        records: list[ReinforcedV2StageRecord] = []
        stage_index = 0
        current_stage_id = self._stage_id(stage_index)
        seed_record = self._save_stage(
            current,
            current_stage_id,
            iteration=stage_index,
            proposal_index=0,
            parent_stage_id=None,
            mutation_operator="seed",
            mutation_details="initial sampled source baseline",
        )
        records.append(seed_record)

        stagnation = 0
        accepted_ops: Counter = Counter()
        halt_reason = "max_proposals"
        proposals_attempted = 0

        for proposal_index in range(1, cfg.max_proposals + 1):
            proposals_attempted = proposal_index
            best_candidate = None
            best_payload = None
            best_operator = "none"
            best_details = ""
            proposal_guidance = self._build_batch_guidance(current.sequences, current_tensor_state)
            mutation_pool: list[tuple[MutationPayload, str, str, float]] = []

            # Precompute token/bigram counts once per proposal (shared across
            # all n_candidates calls — no recomputation until next acceptance).
            if self._scoring_state is not None:
                proposal_token_counts: Counter | None = self._scoring_state.token_counts
                proposal_bigram_counts: Counter | None = self._scoring_state.word_bigram_counts
            else:
                proposal_token_counts = None
                proposal_bigram_counts = None

            for _ in range(cfg.n_candidates):
                mutated, operator, details, mutation_cost = self._mutate_candidate(
                    current.sequences,
                    rng,
                    guidance=proposal_guidance,
                    precomputed_token_counts=proposal_token_counts,
                    precomputed_bigram_counts=proposal_bigram_counts,
                )
                if mutated is None:
                    continue
                mutation_pool.append((mutated, operator, details, mutation_cost))

            if not mutation_pool:
                self._emit_live_event(
                    proposal_index=proposal_index,
                    candidate_index=0,
                    outcome="no_valid_mutation",
                    operator="none",
                    details="no valid mutation generated",
                    mutation_cost=0.0,
                )
                stagnation += 1
                if stagnation >= cfg.patience:
                    halt_reason = "no_valid_mutations"
                    break
                continue

            scored_pool = self._score_mutation_pool(
                current=current,
                mutation_pool=mutation_pool,
                proposal_guidance=proposal_guidance,
            )

            best_index = -1
            for idx, (candidate, operator, details, _, payload) in enumerate(scored_pool, start=1):
                if best_candidate is None or candidate.total_score > best_candidate.total_score:
                    best_candidate = candidate
                    best_payload = payload
                    best_operator = operator
                    best_details = details
                    best_index = idx

            improvement = best_candidate.total_score - current.total_score
            accepted = improvement > cfg.min_improvement

            for idx, (candidate, operator, details, mutation_cost, _) in enumerate(scored_pool, start=1):
                outcome = "accepted" if accepted and idx == best_index else "best_rejected" if idx == best_index else "rejected"
                self._emit_live_event(
                    proposal_index=proposal_index,
                    candidate_index=idx,
                    outcome=outcome,
                    operator=operator,
                    details=details,
                    mutation_cost=mutation_cost,
                    candidate=candidate,
                    score_delta=candidate.total_score - current.total_score,
                    backend=str(candidate.diagnostics.get("candidate_scoring_backend", "python")),
                )

            if improvement > cfg.min_improvement:
                stage_index += 1
                parent_stage_id = current_stage_id
                if best_payload is not None and best_payload.is_sparse:
                    best_candidate.sequences = best_payload.materialize(current.sequences)
                current = best_candidate
                tensor_update = None
                if current_tensor_state is not None:
                    tensor_update = current_tensor_state.apply_sequences(current.sequences)
                if self._scoring_state is not None:
                    self._scoring_state.commit(current.sequences)
                if proposal_guidance is None:
                    current.diagnostics = {
                        **current.diagnostics,
                        "batch_guidance_backend": "python_only",
                        "batch_guidance_batch_size": 0,
                        "batch_guidance_selected_count": 0,
                    }
                if tensor_update is not None:
                    current.diagnostics = {
                        **current.diagnostics,
                        "batch_guidance_tensor_state_update_mode": tensor_update.mode,
                        "batch_guidance_tensor_anchor_vocab_size": tensor_update.anchor_vocab_size,
                        "batch_guidance_tensor_oov_tokens": list(tensor_update.oov_tokens),
                    }
                self._annotate_alignment(current)
                current_stage_id = self._stage_id(stage_index)
                record = self._save_stage(
                    current,
                    current_stage_id,
                    iteration=stage_index,
                    proposal_index=proposal_index,
                    parent_stage_id=parent_stage_id,
                    mutation_operator=best_operator,
                    mutation_details=best_details,
                )
                records.append(record)
                accepted_ops[best_operator] += 1
                stagnation = 0
                self._emit_live_event(
                    proposal_index=proposal_index,
                    candidate_index=best_index,
                    outcome="stage_committed",
                    operator=best_operator,
                    details=best_details,
                    mutation_cost=float(current.mutation_cost),
                    candidate=current,
                    score_delta=improvement,
                    stage_id=current_stage_id,
                    backend=str(current.diagnostics.get("candidate_scoring_backend", "python")),
                )

                log.info(
                    "V4 %s: total=%.4f struct=%.4f form=%.4f align=%.4f weird=%.4f op=%s",
                    current_stage_id,
                    current.total_score,
                    current.latin_structural_score,
                    current.latin_form_score,
                    current.diagnostics.get("family_alignment_score", 0.0),
                    current.diagnostics.get("weirdness_level", 0.0),
                    best_operator,
                )

                if stage_index + 1 >= cfg.max_accepted_stages:
                    halt_reason = "max_accepted_stages"
                    break
            else:
                stagnation += 1
                if stagnation >= cfg.patience:
                    halt_reason = "stable"
                    if records:
                        records[-1].flags.append("stable")
                        records[-1].save(self.records_dir / f"{records[-1].stage_id}.json")
                    break

        self._flush_live_events()
        self._scoring_state = None  # release memory after run
        self._save_summary(records, cfg, proposals_attempted, halt_reason, accepted_ops)
        return records

    def _save_summary(
        self,
        records: list[ReinforcedV2StageRecord],
        cfg: ReinforcedV4Config,
        proposals_attempted: int,
        halt_reason: str,
        accepted_ops: Counter,
    ) -> None:
        best_record = max(records, key=lambda r: r.total_score) if records else None
        summary = {
            "language": self.language,
            "algorithm": "relational_v4",
            "config": cfg.to_dict(),
            "total_stages": len(records),
            "accepted_mutation_stages": max(len(records) - 1, 0),
            "proposals_attempted": proposals_attempted,
            "halt_reason": halt_reason,
            "accepted_operator_counts": dict(accepted_ops),
            "final_stage_id": records[-1].stage_id if records else None,
            "final_total_score": records[-1].total_score if records else None,
            "final_latin_structural_score": records[-1].latin_structural_score if records else None,
            "final_latin_form_score": records[-1].latin_form_score if records else None,
            "final_family_alignment_score": records[-1].diagnostics.get("family_alignment_score") if records else None,
            "final_weirdness_level": records[-1].diagnostics.get("weirdness_level") if records else None,
            "final_coherence_label": records[-1].diagnostics.get("coherence_label") if records else None,
            "best_stage_id": best_record.stage_id if best_record else None,
            "best_total_score": best_record.total_score if best_record else None,
            "best_latin_structural_score": best_record.latin_structural_score if best_record else None,
            "best_latin_form_score": best_record.latin_form_score if best_record else None,
            "best_family_alignment_score": best_record.diagnostics.get("family_alignment_score") if best_record else None,
            "best_corpus_json": best_record.artifact_paths.get("corpus_json") if best_record else None,
            "best_preview_txt": best_record.artifact_paths.get("preview_txt") if best_record else None,
            "stages": [r.to_dict() for r in records],
        }
        path = self.output_dir / "run_summary.json"
        with path.open("w", encoding="utf-8") as fh:
            json.dump(summary, fh, ensure_ascii=False, indent=2)
        log.info("Saved reinforced v4 summary to %s", path)


def run(
    language: str,
    config: ReinforcedV4Config | None = None,
    input_path: Path | None = None,
) -> list[ReinforcedV2StageRecord]:
    if input_path is None:
        input_path = PROCESSED_DIR / "romance" / f"{language}_tokens.json"

    log.info("Loading source corpus from %s", input_path)
    with input_path.open(encoding="utf-8") as fh:
        corpus = json.load(fh)
    sequences = corpus["sequences"]
    log.info("Loaded %d sequences", len(sequences))

    cfg = config or ReinforcedV4Config()
    latin_structural_ref = LatinReference()
    latin_form_ref = LatinFormReference()
    references = ReferenceSet()

    engine = RelationalReinforcedRetrodictionEngineV4(
        language=language,
        source_sequences=sequences,
        latin_structural_ref=latin_structural_ref,
        latin_form_ref=latin_form_ref,
        config=cfg,
        references=references,
    )
    return engine.run()
