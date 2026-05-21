"""
Relational Reinforced Retrodiction V3
=====================================
Purpose:
    Experimental successor to v2 with stranger bundled mutations and stronger
    Latin-side reward amplification for genuinely good moves.
"""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.retrodiction.engine_reinforced import LANG_CODES, LatinReference
from src.retrodiction.engine_reinforced_v2 import (
    CandidateState,
    LatinFormReference,
    OPERATOR_NAMES as V2_OPERATOR_NAMES,
    ReinforcedV2Config,
    ReinforcedV2StageRecord,
    RelationalReinforcedRetrodictionEngine,
)
from src.retrodiction.similarity import ReferenceSet

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RETRODICTION_DIR = PROJECT_ROOT / "data" / "retrodiction"

V3_OPERATOR_NAMES = V2_OPERATOR_NAMES + (
    "function_word_burst",
    "paradigm_family_rewrite",
    "macro_bundle_rewrite",
)


@dataclass
class ReinforcedV3Config(ReinforcedV2Config):
    """Experimental v3 configuration with stronger reward amplification."""

    max_proposals: int = 120
    max_accepted_stages: int = 24
    patience: int = 20
    n_candidates: int = 8
    min_improvement: float = 0.0001

    function_burst_min_tokens: int = 2
    function_burst_max_tokens: int = 5
    paradigm_prefix_min_len: int = 3
    paradigm_prefix_max_len: int = 5
    macro_bundle_min_steps: int = 2
    macro_bundle_max_steps: int = 4

    reward_struct_gain_weight: float = 8.0
    reward_form_gain_weight: float = 4.0
    reward_suffix_gain_weight: float = 2.0
    reward_trigram_gain_weight: float = 2.0
    reward_joint_bonus: float = 0.01
    reward_penalty_relief: float = 1.0
    reward_max_coherence_drop: float = 0.05

    operator_weights: tuple[float, ...] = (
        0.15,  # token_char_edit
        0.13,  # suffix_family_rewrite
        0.06,  # swap_bigram_order
        0.07,  # split_token
        0.06,  # merge_bigram
        0.14,  # sequence_span_rewrite
        0.13,  # function_word_burst
        0.13,  # paradigm_family_rewrite
        0.13,  # macro_bundle_rewrite
    )

    def to_dict(self) -> dict:
        data = super().to_dict()
        data.update(
            {
                "function_burst_min_tokens": self.function_burst_min_tokens,
                "function_burst_max_tokens": self.function_burst_max_tokens,
                "paradigm_prefix_min_len": self.paradigm_prefix_min_len,
                "paradigm_prefix_max_len": self.paradigm_prefix_max_len,
                "macro_bundle_min_steps": self.macro_bundle_min_steps,
                "macro_bundle_max_steps": self.macro_bundle_max_steps,
                "reward_struct_gain_weight": self.reward_struct_gain_weight,
                "reward_form_gain_weight": self.reward_form_gain_weight,
                "reward_suffix_gain_weight": self.reward_suffix_gain_weight,
                "reward_trigram_gain_weight": self.reward_trigram_gain_weight,
                "reward_joint_bonus": self.reward_joint_bonus,
                "reward_penalty_relief": self.reward_penalty_relief,
                "reward_max_coherence_drop": self.reward_max_coherence_drop,
            }
        )
        return data


class RelationalReinforcedRetrodictionEngineV3(RelationalReinforcedRetrodictionEngine):
    """
    v3 search adds weirder mutation operators and amplifies Latin-consistent gains.
    """

    def __init__(
        self,
        language: str,
        source_sequences: list[list[str]],
        latin_structural_ref: LatinReference,
        latin_form_ref: LatinFormReference,
        config: ReinforcedV3Config | None = None,
        output_dir: Path | None = None,
        references: ReferenceSet | None = None,
    ) -> None:
        cfg = config or ReinforcedV3Config()
        if output_dir is None:
            output_dir = RETRODICTION_DIR / language / "v3"
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

    def _stage_id(self, iteration: int) -> str:
        return f"{self.lang_code}_v3_{iteration:03d}"

    def _choose_operator(self, rng: np.random.Generator) -> str:
        weights = np.array(self.config.operator_weights, dtype=np.float64)
        weights /= weights.sum()
        idx = int(rng.choice(len(V3_OPERATOR_NAMES), p=weights))
        return V3_OPERATOR_NAMES[idx]

    def _weirdify_token_form(self, tok: str, rng: np.random.Generator) -> str:
        best = tok
        best_score = self.latin_form_ref.score_token(tok)
        candidate = tok

        n_rounds = int(rng.integers(2, 5))
        for _ in range(n_rounds):
            candidate = self._random_edit_token_form(candidate, rng)
            score = self.latin_form_ref.score_token(candidate)
            if score > best_score and len(candidate) >= 2:
                best = candidate
                best_score = score

        return best

    def _mutate_function_word_burst(
        self,
        sequences: list[list[str]],
        token_counts: Counter,
        rng: np.random.Generator,
    ) -> tuple[list[list[str]] | None, str, float]:
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

        new_sequences, affected = self._apply_token_rewrite(sequences, rewrite_map)
        if affected == 0:
            return None, "function-word burst affected nothing", 0.0
        cost = 0.55 + 0.04 * len(rewrite_map)
        return new_sequences, f"{len(rewrite_map)} short tokens: {', '.join(details_parts[:5])}", cost

    def _mutate_paradigm_family_rewrite(
        self,
        sequences: list[list[str]],
        token_counts: Counter,
        rng: np.random.Generator,
    ) -> tuple[list[list[str]] | None, str, float]:
        families: dict[str, list[str]] = defaultdict(list)
        for tok in token_counts:
            if len(tok) < 5:
                continue
            for prefix_len in range(self.config.paradigm_prefix_min_len, self.config.paradigm_prefix_max_len + 1):
                if len(tok) > prefix_len + 1:
                    families[tok[:prefix_len]].append(tok)

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

        new_sequences, affected = self._apply_token_rewrite(sequences, rewrite_map)
        if affected == 0:
            return None, f"family {prefix} affected nothing", 0.0
        cost = 0.75 + 0.02 * len(rewrite_map)
        return new_sequences, f"prefix {prefix} across {len(rewrite_map)} types: {', '.join(details_parts[:4])}", cost

    def _apply_named_operator(
        self,
        operator: str,
        sequences: list[list[str]],
        token_counts: Counter,
        bigram_counts: Counter,
        rng: np.random.Generator,
    ) -> tuple[list[list[str]] | None, str, float]:
        if operator == "token_char_edit":
            return self._mutate_token_char_edit(sequences, token_counts, rng)
        if operator == "suffix_family_rewrite":
            return self._mutate_suffix_family(sequences, token_counts, rng)
        if operator == "swap_bigram_order":
            return self._mutate_swap_bigram_order(sequences, bigram_counts, rng)
        if operator == "split_token":
            return self._mutate_split_token(sequences, token_counts, rng)
        if operator == "merge_bigram":
            return self._mutate_merge_bigram(sequences, bigram_counts, rng)
        if operator == "sequence_span_rewrite":
            return self._mutate_sequence_span_rewrite(sequences, rng)
        if operator == "function_word_burst":
            return self._mutate_function_word_burst(sequences, token_counts, rng)
        if operator == "paradigm_family_rewrite":
            return self._mutate_paradigm_family_rewrite(sequences, token_counts, rng)
        raise ValueError(f"Unknown operator: {operator}")

    def _mutate_macro_bundle_rewrite(
        self,
        sequences: list[list[str]],
        rng: np.random.Generator,
    ) -> tuple[list[list[str]] | None, str, float]:
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

        mutated = self._clone_sequences(sequences)
        details_parts: list[str] = []
        total_cost = 0.6

        n_steps = int(
            rng.integers(
                self.config.macro_bundle_min_steps,
                self.config.macro_bundle_max_steps + 1,
            )
        )
        for _ in range(n_steps):
            token_counts = self._token_counts(mutated)
            bigram_counts = self._bigram_counts(mutated)
            operator = str(rng.choice(sub_ops, p=weights))
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
            total_cost += cost
            details_parts.append(f"{operator}:{details}")

        if mutated == sequences:
            return None, "macro bundle produced no change", 0.0

        return mutated, "; ".join(details_parts[:5]), total_cost

    def _mutate_candidate(
        self,
        sequences: list[list[str]],
        rng: np.random.Generator,
    ) -> tuple[list[list[str]] | None, str, str, float]:
        token_counts = self._token_counts(sequences)
        bigram_counts = self._bigram_counts(sequences)

        for _ in range(10):
            operator = self._choose_operator(rng)
            if operator == "macro_bundle_rewrite":
                mutated, details, cost = self._mutate_macro_bundle_rewrite(sequences, rng)
            else:
                mutated, details, cost = self._apply_named_operator(
                    operator,
                    sequences,
                    token_counts,
                    bigram_counts,
                    rng,
                )

            if mutated is not None:
                return mutated, operator, details, cost

        return None, "none", "no valid mutation generated", 0.0

    def _amplify_reward(
        self,
        current: CandidateState,
        candidate: CandidateState,
    ) -> CandidateState:
        struct_gain = candidate.latin_structural_score - current.latin_structural_score
        form_gain = candidate.latin_form_score - current.latin_form_score
        suffix_gain = (
            candidate.form_details.get("latin_suffix_cosine", 0.0)
            - current.form_details.get("latin_suffix_cosine", 0.0)
        )
        trigram_gain = (
            candidate.form_details.get("latin_char_trigram_cosine", 0.0)
            - current.form_details.get("latin_char_trigram_cosine", 0.0)
        )
        coherence_gain = (
            candidate.diagnostics.get("language_likeness_margin", 0.0)
            - current.diagnostics.get("language_likeness_margin", 0.0)
        )

        reward_bonus = 0.0
        reward_bonus += self.config.reward_struct_gain_weight * max(0.0, struct_gain)
        reward_bonus += self.config.reward_form_gain_weight * max(0.0, form_gain)
        reward_bonus += self.config.reward_suffix_gain_weight * max(0.0, suffix_gain)
        reward_bonus += self.config.reward_trigram_gain_weight * max(0.0, trigram_gain)

        penalty_relief = 0.0
        if (
            struct_gain > 0.0
            and form_gain > 0.0
            and coherence_gain >= -self.config.reward_max_coherence_drop
        ):
            reward_bonus += self.config.reward_joint_bonus
            penalty_relief = (
                self.config.mutation_cost_weight
                * candidate.mutation_cost
                * self.config.reward_penalty_relief
            )

        candidate.total_score = float(candidate.total_score + reward_bonus + penalty_relief)
        candidate.diagnostics = {
            **candidate.diagnostics,
            "reward_struct_gain": struct_gain,
            "reward_form_gain": form_gain,
            "reward_suffix_gain": suffix_gain,
            "reward_trigram_gain": trigram_gain,
            "reward_coherence_gain": coherence_gain,
            "reward_bonus": reward_bonus,
            "reward_penalty_relief": penalty_relief,
            "reward_effective_total_score": candidate.total_score,
        }
        return candidate

    def run(self) -> list[ReinforcedV2StageRecord]:
        cfg = self.config
        rng = np.random.default_rng(cfg.seed)

        log.info(
            "Relational reinforced v3: language=%s, num_sequences=%d, proposals=%d, candidates=%d",
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
            best_candidate: CandidateState | None = None
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

            if best_candidate is None:
                stagnation += 1
                if stagnation >= cfg.patience:
                    halt_reason = "no_valid_mutations"
                    break
                continue

            improvement = best_candidate.total_score - current.total_score
            if improvement > cfg.min_improvement:
                stage_index += 1
                parent_stage_id = current_stage_id
                current = best_candidate
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

                log.info(
                    "V3 %s: total=%.4f struct=%.4f form=%.4f bonus=%.4f op=%s",
                    current_stage_id,
                    current.total_score,
                    current.latin_structural_score,
                    current.latin_form_score,
                    current.diagnostics.get("reward_bonus", 0.0),
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

        self._save_summary(records, cfg, proposals_attempted, halt_reason, accepted_ops)
        return records

    def _save_summary(
        self,
        records: list[ReinforcedV2StageRecord],
        cfg: ReinforcedV3Config,
        proposals_attempted: int,
        halt_reason: str,
        accepted_ops: Counter,
    ) -> None:
        best_record = max(records, key=lambda r: r.total_score) if records else None
        summary = {
            "language": self.language,
            "algorithm": "relational_v3",
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
            "final_coherence_label": records[-1].diagnostics.get("coherence_label") if records else None,
            "best_stage_id": best_record.stage_id if best_record else None,
            "best_total_score": best_record.total_score if best_record else None,
            "best_latin_structural_score": best_record.latin_structural_score if best_record else None,
            "best_latin_form_score": best_record.latin_form_score if best_record else None,
            "best_corpus_json": best_record.artifact_paths.get("corpus_json") if best_record else None,
            "best_preview_txt": best_record.artifact_paths.get("preview_txt") if best_record else None,
            "stages": [r.to_dict() for r in records],
        }
        path = self.output_dir / "run_summary.json"
        with path.open("w", encoding="utf-8") as fh:
            json.dump(summary, fh, ensure_ascii=False, indent=2)
        log.info("Saved reinforced v3 summary to %s", path)


def run(
    language: str,
    config: ReinforcedV3Config | None = None,
    input_path: Path | None = None,
) -> list[ReinforcedV2StageRecord]:
    if input_path is None:
        input_path = PROCESSED_DIR / "romance" / f"{language}_tokens.json"

    log.info("Loading source corpus from %s", input_path)
    with input_path.open(encoding="utf-8") as fh:
        corpus = json.load(fh)
    sequences = corpus["sequences"]
    log.info("Loaded %d sequences", len(sequences))

    cfg = config or ReinforcedV3Config()
    latin_structural_ref = LatinReference()
    latin_form_ref = LatinFormReference()
    references = ReferenceSet()

    engine = RelationalReinforcedRetrodictionEngineV3(
        language=language,
        source_sequences=sequences,
        latin_structural_ref=latin_structural_ref,
        latin_form_ref=latin_form_ref,
        config=cfg,
        references=references,
    )
    return engine.run()
