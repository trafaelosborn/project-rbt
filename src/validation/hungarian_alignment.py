"""
Hungarian Family Alignment Diagnostic
====================================
Purpose:
    Diagnostic-only Phase 1 for the proposed v4 control loop.

    This module does NOT change mutation behavior. It measures how globally
    aligned a bridge corpus is with Latin by matching mutable family features
    under a Hungarian assignment.

Usage:
    python -m src.validation.hungarian_alignment ^
        --run-summary data/retrodiction/french/v2_convergence/run_summary.json
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.retrodiction.engine_reinforced_v2 import (
    SUFFIX_LEN,
    _build_sparse_profile,
    _sparse_profile_cosine,
)
from src.sequester.guard import load_sequestered, lock_sequestration, unlock_sequestration
from src.validation.checkpoint_compare import default_checkpoint_stage_ids

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VALIDATION_DIR = PROJECT_ROOT / "data" / "validation"

LATIN_ALIGNMENT_UNLOCK_REASON = (
    "Phase 5 validation: Hungarian family alignment diagnostic against "
    "sequestered Latin reference for reinforced bridge evaluation."
)

EPSILON = 1e-9


@dataclass
class FamilyAlignmentConfig:
    max_suffix_families: int = 12
    max_prefix_families: int = 12
    max_short_token_families: int = 12
    prefix_len: int = 3
    suffix_len: int = SUFFIX_LEN
    min_family_types: int = 2
    min_short_token_count: int = 3
    short_token_min_len: int = 2
    short_token_max_len: int = 4
    max_latin_sequences: int = 50_000

    mass_weight: float = 0.20
    trigram_weight: float = 0.45
    suffix_weight: float = 0.25
    length_weight: float = 0.10
    kind_mismatch_penalty: float = 0.20
    unmatched_penalty: float = 1.00
    top_pairs_to_keep: int = 8

    def to_dict(self) -> dict:
        return {
            "max_suffix_families": self.max_suffix_families,
            "max_prefix_families": self.max_prefix_families,
            "max_short_token_families": self.max_short_token_families,
            "prefix_len": self.prefix_len,
            "suffix_len": self.suffix_len,
            "min_family_types": self.min_family_types,
            "min_short_token_count": self.min_short_token_count,
            "short_token_min_len": self.short_token_min_len,
            "short_token_max_len": self.short_token_max_len,
            "max_latin_sequences": self.max_latin_sequences,
            "mass_weight": self.mass_weight,
            "trigram_weight": self.trigram_weight,
            "suffix_weight": self.suffix_weight,
            "length_weight": self.length_weight,
            "kind_mismatch_penalty": self.kind_mismatch_penalty,
            "unmatched_penalty": self.unmatched_penalty,
            "top_pairs_to_keep": self.top_pairs_to_keep,
        }


@dataclass
class FamilyFeature:
    family_id: str
    kind: str
    mass: float
    total_occurrences: int
    member_token_count: int
    mean_token_length: float
    sample_tokens: list[str]
    char_trigram_profile: dict[str, float]
    suffix_profile: dict[str, float]

    def to_dict(self) -> dict:
        return {
            "family_id": self.family_id,
            "kind": self.kind,
            "mass": round(self.mass, 6),
            "total_occurrences": self.total_occurrences,
            "member_token_count": self.member_token_count,
            "mean_token_length": round(self.mean_token_length, 6),
            "sample_tokens": self.sample_tokens,
        }


@dataclass
class FamilyInventory:
    label: str
    families: list[FamilyFeature]
    total_tokens: int


def _char_trigram_profile_from_token_counts(token_counts: dict[str, int]) -> dict[str, float]:
    counter: Counter = Counter()
    for tok, count in token_counts.items():
        text = f"^{tok}$"
        if len(text) < 3:
            continue
        for i in range(len(text) - 2):
            counter[text[i : i + 3]] += int(count)
    return _build_sparse_profile(counter, top_n=2500)


def _suffix_profile_from_token_counts(token_counts: dict[str, int], suffix_len: int) -> dict[str, float]:
    counter: Counter = Counter()
    for tok, count in token_counts.items():
        if len(tok) >= suffix_len:
            counter[tok[-suffix_len:]] += int(count)
    return _build_sparse_profile(counter, top_n=800)


def _make_family(
    family_id: str,
    kind: str,
    member_token_counts: dict[str, int],
    total_tokens: int,
    suffix_len: int,
) -> FamilyFeature:
    total_occurrences = int(sum(member_token_counts.values()))
    member_token_count = len(member_token_counts)
    sample_tokens = [tok for tok, _ in sorted(member_token_counts.items(), key=lambda item: (-item[1], item[0]))[:5]]
    mean_token_length = (
        sum(len(tok) * count for tok, count in member_token_counts.items()) / max(total_occurrences, 1)
    )
    return FamilyFeature(
        family_id=family_id,
        kind=kind,
        mass=float(total_occurrences / max(total_tokens, 1)),
        total_occurrences=total_occurrences,
        member_token_count=member_token_count,
        mean_token_length=float(mean_token_length),
        sample_tokens=sample_tokens,
        char_trigram_profile=_char_trigram_profile_from_token_counts(member_token_counts),
        suffix_profile=_suffix_profile_from_token_counts(member_token_counts, suffix_len=suffix_len),
    )


def extract_family_inventory(
    label: str,
    sequences: list[list[str]],
    config: FamilyAlignmentConfig | None = None,
) -> FamilyInventory:
    cfg = config or FamilyAlignmentConfig()
    token_counts = Counter(tok for seq in sequences for tok in seq if tok)
    total_tokens = int(sum(token_counts.values()))

    families: list[FamilyFeature] = []

    suffix_groups: dict[str, dict[str, int]] = defaultdict(dict)
    for tok, count in token_counts.items():
        if len(tok) < cfg.suffix_len + 1:
            continue
        suffix_groups[tok[-cfg.suffix_len :]][tok] = int(count)

    suffix_candidates = []
    for suffix, members in suffix_groups.items():
        if len(members) < cfg.min_family_types:
            continue
        suffix_candidates.append((suffix, sum(members.values()), members))
    suffix_candidates.sort(key=lambda item: (-item[1], item[0]))
    for suffix, _, members in suffix_candidates[: cfg.max_suffix_families]:
        families.append(
            _make_family(
                family_id=f"suffix:{suffix}",
                kind="suffix",
                member_token_counts=members,
                total_tokens=total_tokens,
                suffix_len=cfg.suffix_len,
            )
        )

    prefix_groups: dict[str, dict[str, int]] = defaultdict(dict)
    for tok, count in token_counts.items():
        if len(tok) <= cfg.prefix_len + 1:
            continue
        prefix_groups[tok[: cfg.prefix_len]][tok] = int(count)

    prefix_candidates = []
    for prefix, members in prefix_groups.items():
        if len(members) < cfg.min_family_types:
            continue
        prefix_candidates.append((prefix, sum(members.values()), members))
    prefix_candidates.sort(key=lambda item: (-item[1], item[0]))
    for prefix, _, members in prefix_candidates[: cfg.max_prefix_families]:
        families.append(
            _make_family(
                family_id=f"prefix:{prefix}",
                kind="prefix",
                member_token_counts=members,
                total_tokens=total_tokens,
                suffix_len=cfg.suffix_len,
            )
        )

    short_candidates = [
        (tok, count)
        for tok, count in token_counts.items()
        if cfg.short_token_min_len <= len(tok) <= cfg.short_token_max_len and count >= cfg.min_short_token_count
    ]
    short_candidates.sort(key=lambda item: (-item[1], item[0]))
    for tok, count in short_candidates[: cfg.max_short_token_families]:
        families.append(
            _make_family(
                family_id=f"short:{tok}",
                kind="short_token",
                member_token_counts={tok: int(count)},
                total_tokens=total_tokens,
                suffix_len=cfg.suffix_len,
            )
        )

    return FamilyInventory(label=label, families=families, total_tokens=total_tokens)


def _family_cost(
    a: FamilyFeature,
    b: FamilyFeature,
    config: FamilyAlignmentConfig,
) -> float:
    mass_cost = abs(a.mass - b.mass)
    trigram_cost = 1.0 - _sparse_profile_cosine(a.char_trigram_profile, b.char_trigram_profile)
    suffix_cost = 1.0 - _sparse_profile_cosine(a.suffix_profile, b.suffix_profile)
    length_cost = abs(a.mean_token_length - b.mean_token_length) / max(
        a.mean_token_length,
        b.mean_token_length,
        1.0,
    )
    kind_penalty = config.kind_mismatch_penalty if a.kind != b.kind else 0.0

    total = (
        config.mass_weight * mass_cost
        + config.trigram_weight * trigram_cost
        + config.suffix_weight * suffix_cost
        + config.length_weight * length_cost
        + kind_penalty
    )
    return float(min(max(total, 0.0), 1.0))


def hungarian_alignment_diagnostics(
    inventory: FamilyInventory,
    reference_inventory: FamilyInventory,
    config: FamilyAlignmentConfig | None = None,
) -> dict:
    cfg = config or FamilyAlignmentConfig()
    families = inventory.families
    ref_families = reference_inventory.families

    if not families or not ref_families:
        return {
            "family_alignment_score": 0.0,
            "family_alignment_cost": 1.0,
            "matched_family_count": 0,
            "unmatched_bridge_families": len(families),
            "unmatched_reference_families": len(ref_families),
            "matched_pairs": [],
        }

    cost_matrix = np.zeros((len(families), len(ref_families)), dtype=np.float64)
    for i, fam in enumerate(families):
        for j, ref in enumerate(ref_families):
            cost_matrix[i, j] = _family_cost(fam, ref, cfg)

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matched_pairs = []
    matched_cost_total = 0.0
    matched_rows = set()
    matched_cols = set()
    for r, c in zip(row_ind.tolist(), col_ind.tolist()):
        fam = families[r]
        ref = ref_families[c]
        cost = float(cost_matrix[r, c])
        matched_cost_total += cost
        matched_rows.add(r)
        matched_cols.add(c)
        matched_pairs.append(
            {
                "bridge_family_id": fam.family_id,
                "bridge_kind": fam.kind,
                "bridge_mass": round(fam.mass, 6),
                "bridge_sample_tokens": fam.sample_tokens,
                "reference_family_id": ref.family_id,
                "reference_kind": ref.kind,
                "reference_mass": round(ref.mass, 6),
                "reference_sample_tokens": ref.sample_tokens,
                "pair_cost": round(cost, 6),
            }
        )

    unmatched_bridge = len(families) - len(matched_rows)
    unmatched_ref = len(ref_families) - len(matched_cols)
    total_slots = max(len(families), len(ref_families), 1)
    total_cost = matched_cost_total + cfg.unmatched_penalty * (unmatched_bridge + unmatched_ref)
    normalized_cost = float(total_cost / total_slots)
    alignment_score = float(max(0.0, 1.0 - normalized_cost))

    matched_pairs.sort(key=lambda item: item["pair_cost"])
    worst_pairs = sorted(matched_pairs, key=lambda item: item["pair_cost"], reverse=True)[: cfg.top_pairs_to_keep]

    return {
        "family_alignment_score": alignment_score,
        "family_alignment_cost": normalized_cost,
        "matched_family_count": len(matched_pairs),
        "unmatched_bridge_families": unmatched_bridge,
        "unmatched_reference_families": unmatched_ref,
        "bridge_family_count": len(families),
        "reference_family_count": len(ref_families),
        "best_pairs": matched_pairs[: cfg.top_pairs_to_keep],
        "worst_pairs": worst_pairs,
    }


def _load_sequences_from_corpus_json(path: Path) -> list[list[str]]:
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    return data["sequences"]


def load_latin_family_reference(
    config: FamilyAlignmentConfig | None = None,
    unlock_reason: str = LATIN_ALIGNMENT_UNLOCK_REASON,
) -> FamilyInventory:
    cfg = config or FamilyAlignmentConfig()
    unlock_sequestration(unlock_reason)
    try:
        corpus = load_sequestered("latin")
    finally:
        lock_sequestration()
    sequences = corpus["sequences"][: cfg.max_latin_sequences]
    return extract_family_inventory("latin", sequences, cfg)


def compare_run_to_reference(
    run_summary_path: Path,
    reference_inventory: FamilyInventory,
    checkpoint_stage_ids: list[str] | None = None,
    output_path: Path | None = None,
    config: FamilyAlignmentConfig | None = None,
) -> dict:
    cfg = config or FamilyAlignmentConfig()
    with run_summary_path.open(encoding="utf-8") as fh:
        run_summary = json.load(fh)

    if checkpoint_stage_ids is None:
        checkpoint_stage_ids = default_checkpoint_stage_ids(run_summary)

    stage_by_id = {stage["stage_id"]: stage for stage in run_summary.get("stages", [])}
    missing = [stage_id for stage_id in checkpoint_stage_ids if stage_id not in stage_by_id]
    if missing:
        raise ValueError(f"Checkpoint stage ids missing from run summary: {missing}")

    comparisons = []
    for stage_id in checkpoint_stage_ids:
        stage = stage_by_id[stage_id]
        corpus_path = Path(stage["artifacts"]["corpus_json"])
        sequences = _load_sequences_from_corpus_json(corpus_path)
        inventory = extract_family_inventory(stage_id, sequences, cfg)
        diagnostics = hungarian_alignment_diagnostics(inventory, reference_inventory, cfg)
        item = {
            "stage_id": stage["stage_id"],
            "iteration": stage["iteration"],
            "mutation_operator": stage["mutation_operator"],
            "preview_txt": stage["artifacts"].get("preview_txt"),
            "corpus_json": stage["artifacts"].get("corpus_json"),
            "latin_structural_score": stage.get("latin_structural_score"),
            "latin_form_score": stage.get("latin_form_score"),
            "total_score": stage.get("total_score"),
            "coherence_label": stage.get("diagnostics", {}).get("coherence_label"),
            "bridge_families": [fam.to_dict() for fam in inventory.families[: cfg.top_pairs_to_keep]],
            **diagnostics,
        }
        comparisons.append(item)

    best_by_alignment = max(comparisons, key=lambda item: item["family_alignment_score"])
    best_by_lowest_cost = min(comparisons, key=lambda item: item["family_alignment_cost"])

    result = {
        "comparison_date": date.today().isoformat(),
        "run_summary": str(run_summary_path),
        "run_language": run_summary.get("language"),
        "run_algorithm": run_summary.get("algorithm"),
        "reference_label": reference_inventory.label,
        "checkpoint_stage_ids": checkpoint_stage_ids,
        "config": cfg.to_dict(),
        "reference_families": [fam.to_dict() for fam in reference_inventory.families[: cfg.top_pairs_to_keep]],
        "best_by_family_alignment": best_by_alignment,
        "best_by_lowest_cost": best_by_lowest_cost,
        "comparisons": comparisons,
    }

    if output_path is None:
        VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
        run_label = f"{run_summary.get('language', 'run')}_{run_summary_path.parent.name}"
        output_path = VALIDATION_DIR / f"{run_label}_vs_{reference_inventory.label}_family_alignment.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    log.info("Wrote Hungarian family-alignment comparison to %s", output_path)
    return result


def compare_run_to_latin_alignment(
    run_summary_path: Path,
    checkpoint_stage_ids: list[str] | None = None,
    output_path: Path | None = None,
    config: FamilyAlignmentConfig | None = None,
) -> dict:
    cfg = config or FamilyAlignmentConfig()
    latin_inventory = load_latin_family_reference(cfg)
    return compare_run_to_reference(
        run_summary_path=run_summary_path,
        reference_inventory=latin_inventory,
        checkpoint_stage_ids=checkpoint_stage_ids,
        output_path=output_path,
        config=cfg,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hungarian family alignment against Latin")
    parser.add_argument("--run-summary", type=Path, required=True, help="Path to a run_summary.json")
    parser.add_argument(
        "--stage-id",
        dest="stage_ids",
        action="append",
        default=None,
        help="Specific checkpoint stage id to include. Repeat to add more.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path")
    args = parser.parse_args()

    compare_run_to_latin_alignment(
        run_summary_path=args.run_summary,
        checkpoint_stage_ids=args.stage_ids,
        output_path=args.output,
    )
