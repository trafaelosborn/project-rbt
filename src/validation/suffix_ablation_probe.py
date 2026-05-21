"""
Suffix Ablation Probe
=====================

Run a bounded French -> Latin v5 comparison between:

- baseline Latin form scoring
- suffix-off Latin form scoring

This is intended as a lightweight paper-hardening probe rather than a full
production run. It uses the same engine class in both conditions and changes
only the Latin form component weights.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from src.retrodiction.engine_reinforced import LatinReference
from src.retrodiction.engine_reinforced_v2 import LatinFormReference
from src.retrodiction.engine_reinforced_v5 import (
    ReinforcedV5Config,
    RelationalReinforcedRetrodictionEngineV5,
)
from src.retrodiction.similarity import ReferenceSet

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_START_CORPUS = PROJECT_ROOT / "data" / "processed" / "romance" / "french_tokens.json"
VALIDATION_DIR = PROJECT_ROOT / "data" / "validation"


@dataclass(frozen=True)
class ProbeCondition:
    name: str
    char_bigram_weight: float
    char_trigram_weight: float
    suffix_weight: float


CONDITIONS: tuple[ProbeCondition, ...] = (
    ProbeCondition("baseline", 0.40, 0.40, 0.20),
    ProbeCondition("suffix_off", 0.50, 0.50, 0.00),
)


def _load_sequences(path: Path, limit: int) -> list[list[str]]:
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    return [list(seq) for seq in data["sequences"][:limit]]


def _make_config(
    *,
    seed: int,
    max_proposals: int,
    n_candidates: int,
    num_sequences: int,
    use_fortran_cosine: bool,
) -> ReinforcedV5Config:
    return ReinforcedV5Config(
        num_sequences=num_sequences,
        max_proposals=max_proposals,
        max_accepted_stages=512,
        patience=max_proposals,
        seed=seed,
        n_candidates=n_candidates,
        min_improvement=0.0001,
        token_edit_attempts=6,
        suffix_candidate_samples=8,
        span_min_sequences=2,
        span_max_sequences=5,
        span_edit_min=2,
        span_edit_max=4,
        form_weight=0.75,
        coherence_weight=0.05,
        mutation_cost_weight=0.005,
        operator_weights=[0.15, 0.13, 0.06, 0.07, 0.06, 0.14, 0.13, 0.13, 0.13],
        function_burst_min_tokens=2,
        function_burst_max_tokens=5,
        paradigm_prefix_min_len=3,
        paradigm_prefix_max_len=5,
        macro_bundle_min_steps=2,
        macro_bundle_max_steps=4,
        reward_struct_gain_weight=8.0,
        reward_form_gain_weight=4.0,
        reward_suffix_gain_weight=2.0,
        reward_trigram_gain_weight=2.0,
        reward_joint_bonus=0.01,
        reward_penalty_relief=1.0,
        reward_max_coherence_drop=0.05,
        alignment_beta=8.0,
        weirdness_floor=0.15,
        weirdness_ceiling=1.0,
        weird_operator_gain=1.8,
        stable_operator_gain=1.4,
        use_incremental_scoring=True,
        use_fortran_cosine=use_fortran_cosine,
        use_fortran_batch=False,
        save_dense_matrices=False,
        use_semantic_transparency=False,
        transparency_weight=0.0,
        enable_culture_bombs=False,
        live_event_mode="off",
        live_event_buffer_size=8,
    )


def _run_condition(
    condition: ProbeCondition,
    *,
    sequences: list[list[str]],
    structural_ref: LatinReference,
    references: ReferenceSet,
    seed: int,
    max_proposals: int,
    n_candidates: int,
    output_root: Path,
    use_fortran_cosine: bool,
) -> dict:
    output_dir = output_root / condition.name
    output_dir.mkdir(parents=True, exist_ok=True)

    latin_form_ref = LatinFormReference(
        char_bigram_weight=condition.char_bigram_weight,
        char_trigram_weight=condition.char_trigram_weight,
        suffix_weight=condition.suffix_weight,
    )
    cfg = _make_config(
        seed=seed,
        max_proposals=max_proposals,
        n_candidates=n_candidates,
        num_sequences=len(sequences),
        use_fortran_cosine=use_fortran_cosine,
    )
    engine = RelationalReinforcedRetrodictionEngineV5(
        language="french",
        source_sequences=sequences,
        latin_structural_ref=structural_ref,
        latin_form_ref=latin_form_ref,
        config=cfg,
        output_dir=output_dir,
        references=references,
    )
    records = engine.run()
    summary_path = output_dir / "run_summary.json"
    with summary_path.open(encoding="utf-8") as fh:
        summary = json.load(fh)
    summary["record_count"] = len(records)
    summary["condition"] = {
        "name": condition.name,
        "char_bigram_weight": condition.char_bigram_weight,
        "char_trigram_weight": condition.char_trigram_weight,
        "suffix_weight": condition.suffix_weight,
    }
    return summary


def run_probe(
    *,
    start_corpus: Path = DEFAULT_START_CORPUS,
    seed: int = 42,
    max_proposals: int = 500,
    n_candidates: int = 16,
    num_sequences: int = 800,
    use_fortran_cosine: bool = True,
    output_root: Path | None = None,
) -> dict:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = output_root or (VALIDATION_DIR / f"french_v5_suffix_ablation_probe_{stamp}")
    root.mkdir(parents=True, exist_ok=True)

    sequences = _load_sequences(start_corpus, num_sequences)
    structural_ref = LatinReference()
    references = ReferenceSet()

    results = {}
    for condition in CONDITIONS:
        results[condition.name] = _run_condition(
            condition,
            sequences=sequences,
            structural_ref=structural_ref,
            references=references,
            seed=seed,
            max_proposals=max_proposals,
            n_candidates=n_candidates,
            output_root=root,
            use_fortran_cosine=use_fortran_cosine,
        )

    baseline = results["baseline"]
    suffix_off = results["suffix_off"]
    comparison = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "start_corpus": str(start_corpus),
        "seed": seed,
        "max_proposals": max_proposals,
        "n_candidates": n_candidates,
        "num_sequences": num_sequences,
        "use_fortran_cosine": use_fortran_cosine,
        "conditions": results,
        "delta_suffix_off_minus_baseline": {
            "final_total_score": float(suffix_off["final_total_score"] - baseline["final_total_score"]),
            "final_latin_structural_score": float(
                suffix_off["final_latin_structural_score"] - baseline["final_latin_structural_score"]
            ),
            "final_latin_form_score": float(
                suffix_off["final_latin_form_score"] - baseline["final_latin_form_score"]
            ),
            "final_family_alignment_score": float(
                suffix_off["final_family_alignment_score"] - baseline["final_family_alignment_score"]
            ),
            "accepted_mutation_stages": int(
                suffix_off["accepted_mutation_stages"] - baseline["accepted_mutation_stages"]
            ),
        },
    }

    output_path = root / "summary.json"
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(comparison, fh, ensure_ascii=False, indent=2)

    return {
        "output_root": str(root),
        "summary_path": str(output_path),
        "comparison": comparison,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a bounded suffix-profile ablation probe for French -> Latin v5")
    parser.add_argument("--start-corpus", type=Path, default=DEFAULT_START_CORPUS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-proposals", type=int, default=500)
    parser.add_argument("--n-candidates", type=int, default=16)
    parser.add_argument("--num-sequences", type=int, default=800)
    parser.add_argument("--use-fortran-cosine", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()

    result = run_probe(
        start_corpus=args.start_corpus,
        seed=args.seed,
        max_proposals=args.max_proposals,
        n_candidates=args.n_candidates,
        num_sequences=args.num_sequences,
        use_fortran_cosine=args.use_fortran_cosine,
        output_root=args.output_root,
    )
    print(result["summary_path"])


if __name__ == "__main__":
    main()
