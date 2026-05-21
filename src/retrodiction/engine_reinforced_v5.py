"""
Relational Reinforced Retrodiction V5
=====================================
Purpose:
    Experimental successor to v4 that adds plateau-triggered exogenous shocks.

    When the search stalls for a configurable plateau window, v5 proposes a
    structured "culture bomb" bundle instead of stopping immediately.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.retrodiction.engine_reinforced import LatinReference
from src.retrodiction.engine_reinforced_v2 import LatinFormReference, ReinforcedV2StageRecord
from src.retrodiction.engine_reinforced_v4 import (
    RETRODICTION_DIR,
    ReinforcedV4Config,
    RelationalReinforcedRetrodictionEngineV4,
)
from src.retrodiction.similarity import ReferenceSet
from src.validation.hungarian_alignment import FamilyInventory

log = logging.getLogger(__name__)

SHOCK_OPERATOR_NAMES = (
    "sequence_span_rewrite",
    "function_word_burst",
    "paradigm_family_rewrite",
    "macro_bundle_rewrite",
    "split_token",
    "token_char_edit",
)


@dataclass
class ReinforcedV5Config(ReinforcedV4Config):
    """v5 configuration: v4 plus plateau-triggered culture bombs."""

    enable_culture_bombs: bool = False
    shock_plateau_window: int = 10
    max_culture_bombs: int = 10
    culture_bomb_candidates: int = 6
    culture_bomb_min_steps: int = 3
    culture_bomb_max_steps: int = 7
    culture_bomb_cost_discount: float = 0.45

    def to_dict(self) -> dict:
        data = super().to_dict()
        data.update(
            {
                "enable_culture_bombs": self.enable_culture_bombs,
                "shock_plateau_window": self.shock_plateau_window,
                "max_culture_bombs": self.max_culture_bombs,
                "culture_bomb_candidates": self.culture_bomb_candidates,
                "culture_bomb_min_steps": self.culture_bomb_min_steps,
                "culture_bomb_max_steps": self.culture_bomb_max_steps,
                "culture_bomb_cost_discount": self.culture_bomb_cost_discount,
            }
        )
        return data


class RelationalReinforcedRetrodictionEngineV5(RelationalReinforcedRetrodictionEngineV4):
    """
    v5 keeps the v4 controller but injects plateau-triggered shock candidates.
    """

    def __init__(
        self,
        language: str,
        source_sequences: list[list[str]],
        latin_structural_ref: LatinReference,
        latin_form_ref: LatinFormReference,
        config: ReinforcedV5Config | None = None,
        output_dir: Path | None = None,
        references: ReferenceSet | None = None,
        family_reference_inventory: FamilyInventory | None = None,
    ) -> None:
        cfg = config or ReinforcedV5Config()
        if output_dir is None:
            output_dir = RETRODICTION_DIR / language / "v5"
        super().__init__(
            language=language,
            source_sequences=source_sequences,
            latin_structural_ref=latin_structural_ref,
            latin_form_ref=latin_form_ref,
            config=cfg,
            output_dir=output_dir,
            references=references,
            family_reference_inventory=family_reference_inventory,
        )
        self.config = cfg

    def _stage_id(self, iteration: int) -> str:
        return f"{self.lang_code}_v5_{iteration:03d}"

    def _mutate_culture_bomb(
        self,
        sequences: list[list[str]],
        rng: np.random.Generator,
    ) -> tuple[list[list[str]] | None, str, float]:
        weights = np.array([0.20, 0.18, 0.18, 0.18, 0.14, 0.12], dtype=np.float64)
        weights /= weights.sum()

        mutated = self._clone_sequences(sequences)
        details_parts: list[str] = []
        raw_cost = 0.9

        n_steps = int(
            rng.integers(
                self.config.culture_bomb_min_steps,
                self.config.culture_bomb_max_steps + 1,
            )
        )

        for _ in range(n_steps):
            token_counts = self._token_counts(mutated)
            bigram_counts = self._bigram_counts(mutated)
            operator = str(rng.choice(SHOCK_OPERATOR_NAMES, p=weights))
            if operator == "macro_bundle_rewrite":
                next_sequences, details, cost = self._mutate_macro_bundle_rewrite(mutated, rng)
            else:
                next_sequences, details, cost = self._apply_named_operator(
                    operator,
                    mutated,
                    token_counts,
                    bigram_counts,
                    rng,
                )
            if next_sequences is None or next_sequences == mutated:
                continue
            mutated = next_sequences
            raw_cost += cost
            details_parts.append(f"{operator}:{details}")

        if mutated == sequences:
            return None, "culture bomb produced no change", 0.0

        discounted_cost = max(0.35, raw_cost * self.config.culture_bomb_cost_discount)
        detail_text = "; ".join(details_parts[:6]) if details_parts else "shock rewrite"
        return mutated, detail_text, discounted_cost

    def _best_culture_bomb_candidate(
        self,
        current,
        rng: np.random.Generator,
    ):
        best_candidate = None
        best_details = ""
        for _ in range(self.config.culture_bomb_candidates):
            mutated, details, mutation_cost = self._mutate_culture_bomb(current.sequences, rng)
            if mutated is None:
                continue
            candidate = self._evaluate_sequences(mutated, mutation_cost=mutation_cost)
            candidate = self._amplify_reward(current, candidate)
            candidate.operator = "culture_bomb"
            candidate.details = details
            if best_candidate is None or candidate.total_score > best_candidate.total_score:
                best_candidate = candidate
                best_details = details
        return best_candidate, best_details

    def run(self) -> list[ReinforcedV2StageRecord]:
        cfg = self.config
        if not cfg.enable_culture_bombs:
            log.info(
                "Relational reinforced v5 running in plain mode (culture bombs disabled, fortran_batch=%s)",
                cfg.use_fortran_batch,
            )
            return super().run()

        rng = np.random.default_rng(cfg.seed)

        log.info(
            "Relational reinforced v5: language=%s, num_sequences=%d, proposals=%d, candidates=%d, plateau_window=%d",
            self.language, cfg.num_sequences, cfg.max_proposals, cfg.n_candidates, cfg.shock_plateau_window,
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
            "culture_bombs_used": 0,
        }
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
        culture_bombs_used = 0
        accepted_ops: Counter = Counter()
        halt_reason = "max_proposals"
        proposals_attempted = 0

        for proposal_index in range(1, cfg.max_proposals + 1):
            proposals_attempted = proposal_index
            best_candidate = None
            best_operator = "none"
            best_details = ""

            for _ in range(cfg.n_candidates):
                mutated, operator, details, mutation_cost = self._mutate_candidate(current.sequences, rng)
                if mutated is None:
                    continue
                candidate = self._evaluate_sequences(mutated, mutation_cost=mutation_cost)
                candidate = self._amplify_reward(current, candidate)
                candidate.operator = operator
                candidate.details = details
                if best_candidate is None or candidate.total_score > best_candidate.total_score:
                    best_candidate = candidate
                    best_operator = operator
                    best_details = details

            accepted = False
            if best_candidate is not None:
                improvement = best_candidate.total_score - current.total_score
                if improvement > cfg.min_improvement:
                    stage_index += 1
                    parent_stage_id = current_stage_id
                    current = best_candidate
                    self._annotate_alignment(current)
                    current.diagnostics = {
                        **current.diagnostics,
                        "culture_bombs_used": culture_bombs_used,
                    }
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
                    accepted = True

                    log.info(
                        "V5 %s: total=%.4f struct=%.4f form=%.4f align=%.4f weird=%.4f op=%s bombs=%d",
                        current_stage_id,
                        current.total_score,
                        current.latin_structural_score,
                        current.latin_form_score,
                        current.diagnostics.get("family_alignment_score", 0.0),
                        current.diagnostics.get("weirdness_level", 0.0),
                        best_operator,
                        culture_bombs_used,
                    )

                    if stage_index + 1 >= cfg.max_accepted_stages:
                        halt_reason = "max_accepted_stages"
                        break

            if accepted:
                continue

            stagnation += 1
            if stagnation < cfg.shock_plateau_window:
                continue

            if culture_bombs_used >= cfg.max_culture_bombs:
                halt_reason = "shock_budget_exhausted"
                if records:
                    records[-1].flags.append("stable")
                    records[-1].save(self.records_dir / f"{records[-1].stage_id}.json")
                break

            shock_candidate, shock_details = self._best_culture_bomb_candidate(current, rng)
            culture_bombs_used += 1
            if shock_candidate is None:
                halt_reason = "culture_bomb_failed"
                if records:
                    records[-1].flags.append("stable")
                    records[-1].save(self.records_dir / f"{records[-1].stage_id}.json")
                break

            shock_improvement = shock_candidate.total_score - current.total_score
            if shock_improvement <= cfg.min_improvement:
                halt_reason = "culture_bomb_plateau"
                if records:
                    records[-1].flags.append("stable")
                    records[-1].save(self.records_dir / f"{records[-1].stage_id}.json")
                break

            stage_index += 1
            parent_stage_id = current_stage_id
            current = shock_candidate
            self._annotate_alignment(current)
            current.diagnostics = {
                **current.diagnostics,
                "culture_bombs_used": culture_bombs_used,
            }
            current_stage_id = self._stage_id(stage_index)
            record = self._save_stage(
                current,
                current_stage_id,
                iteration=stage_index,
                proposal_index=proposal_index,
                parent_stage_id=parent_stage_id,
                mutation_operator="culture_bomb",
                mutation_details=shock_details,
            )
            records.append(record)
            accepted_ops["culture_bomb"] += 1
            stagnation = 0

            log.info(
                "V5 %s: total=%.4f struct=%.4f form=%.4f align=%.4f weird=%.4f op=culture_bomb bombs=%d",
                current_stage_id,
                current.total_score,
                current.latin_structural_score,
                current.latin_form_score,
                current.diagnostics.get("family_alignment_score", 0.0),
                current.diagnostics.get("weirdness_level", 0.0),
                culture_bombs_used,
            )

            if stage_index + 1 >= cfg.max_accepted_stages:
                halt_reason = "max_accepted_stages"
                break

        self._save_summary(records, cfg, proposals_attempted, halt_reason, accepted_ops, culture_bombs_used)
        return records

    def _save_summary(
        self,
        records: list[ReinforcedV2StageRecord],
        cfg: ReinforcedV5Config,
        proposals_attempted: int,
        halt_reason: str,
        accepted_ops: Counter,
        culture_bombs_used: int = 0,
    ) -> None:
        best_record = max(records, key=lambda r: r.total_score) if records else None
        summary = {
            "language": self.language,
            "algorithm": "relational_v5",
            "config": cfg.to_dict(),
            "total_stages": len(records),
            "accepted_mutation_stages": max(len(records) - 1, 0),
            "proposals_attempted": proposals_attempted,
            "halt_reason": halt_reason,
            "accepted_operator_counts": dict(accepted_ops),
            "culture_bombs_used": culture_bombs_used,
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
        log.info("Saved reinforced v5 summary to %s", path)


def run(
    language: str,
    config: ReinforcedV5Config | None = None,
    input_path: Path | None = None,
) -> list[ReinforcedV2StageRecord]:
    project_root = Path(__file__).resolve().parents[2]
    processed_dir = project_root / "data" / "processed"
    if input_path is None:
        input_path = processed_dir / "romance" / f"{language}_tokens.json"

    log.info("Loading source corpus from %s", input_path)
    with input_path.open(encoding="utf-8") as fh:
        corpus = json.load(fh)
    sequences = corpus["sequences"]
    log.info("Loaded %d sequences", len(sequences))

    cfg = config or ReinforcedV5Config()
    latin_structural_ref = LatinReference()
    latin_form_ref = LatinFormReference()
    references = ReferenceSet()

    engine = RelationalReinforcedRetrodictionEngineV5(
        language=language,
        source_sequences=sequences,
        latin_structural_ref=latin_structural_ref,
        latin_form_ref=latin_form_ref,
        config=cfg,
        references=references,
    )
    return engine.run()
