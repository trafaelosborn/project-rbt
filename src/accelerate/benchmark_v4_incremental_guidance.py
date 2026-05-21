"""
Phase 4 benchmark for the integrated incremental-state v4 guidance path.

This compares the same small v4 run in:
    - python_only mode
    - auto_batch mode with the integrated incremental tensor state
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from src.retrodiction.engine_reinforced import LatinReference
from src.retrodiction.engine_reinforced_v2 import LatinFormReference
from src.retrodiction.engine_reinforced_v4 import (
    ReinforcedV4Config,
    RelationalReinforcedRetrodictionEngineV4,
)
from src.retrodiction.similarity import ReferenceSet
from src.validation.hungarian_alignment import load_latin_family_reference

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "romance"
RETRODICTION_DIR = PROJECT_ROOT / "data" / "retrodiction"


@dataclass(frozen=True)
class ModeBenchmarkResult:
    mode: str
    seconds: float
    accepted_mutation_stages: int
    final_total_score: float | None
    final_latin_structural_score: float | None
    final_latin_form_score: float | None
    final_family_alignment_score: float | None
    batch_guidance_backend: str | None
    tensor_state_update_modes: dict[str, int]

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "seconds": round(self.seconds, 6),
            "accepted_mutation_stages": self.accepted_mutation_stages,
            "final_total_score": self.final_total_score,
            "final_latin_structural_score": self.final_latin_structural_score,
            "final_latin_form_score": self.final_latin_form_score,
            "final_family_alignment_score": self.final_family_alignment_score,
            "batch_guidance_backend": self.batch_guidance_backend,
            "tensor_state_update_modes": dict(self.tensor_state_update_modes),
        }


def _load_sequences(language: str) -> list[list[str]]:
    path = PROCESSED_DIR / f"{language}_tokens.json"
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)["sequences"]


def _run_mode(
    *,
    mode: str,
    language: str,
    sequences: list[list[str]],
    latin_structural_ref: LatinReference,
    latin_form_ref: LatinFormReference,
    references: ReferenceSet,
    family_reference_inventory,
    output_dir: Path,
) -> ModeBenchmarkResult:
    shutil.rmtree(output_dir, ignore_errors=True)

    cfg = ReinforcedV4Config(
        num_sequences=180,
        max_proposals=6,
        max_accepted_stages=4,
        patience=5,
        seed=11,
        n_candidates=4,
        min_improvement=0.00005,
        acceleration_mode=mode,
        acceleration_top_k=128,
        acceleration_max_assignments=12,
        acceleration_hotspot_token_limit=24,
        acceleration_hotspot_pair_limit=12,
    )
    engine = RelationalReinforcedRetrodictionEngineV4(
        language=language,
        source_sequences=sequences,
        latin_structural_ref=latin_structural_ref,
        latin_form_ref=latin_form_ref,
        config=cfg,
        output_dir=output_dir,
        references=references,
        family_reference_inventory=family_reference_inventory,
    )

    started = time.perf_counter()
    records = engine.run()
    seconds = time.perf_counter() - started

    backend = None
    if records:
        backend = records[-1].diagnostics.get("batch_guidance_backend")

    with (output_dir / "run_summary.json").open(encoding="utf-8") as fh:
        summary = json.load(fh)
    tensor_update_modes: dict[str, int] = {}
    for stage in summary.get("stages", []):
        mode_name = stage.get("diagnostics", {}).get("batch_guidance_tensor_state_update_mode")
        if not mode_name:
            continue
        tensor_update_modes[mode_name] = tensor_update_modes.get(mode_name, 0) + 1

    return ModeBenchmarkResult(
        mode=mode,
        seconds=seconds,
        accepted_mutation_stages=int(summary.get("accepted_mutation_stages", 0)),
        final_total_score=summary.get("final_total_score"),
        final_latin_structural_score=summary.get("final_latin_structural_score"),
        final_latin_form_score=summary.get("final_latin_form_score"),
        final_family_alignment_score=summary.get("final_family_alignment_score"),
        batch_guidance_backend=backend,
        tensor_state_update_modes=tensor_update_modes,
    )


def run_benchmark(
    *,
    language: str = "french",
) -> dict:
    sequences = _load_sequences(language)
    latin_structural_ref = LatinReference()
    latin_form_ref = LatinFormReference()
    references = ReferenceSet()
    family_reference_inventory = load_latin_family_reference()

    python_dir = RETRODICTION_DIR / language / "_v4_phase4_bench_python"
    auto_dir = RETRODICTION_DIR / language / "_v4_phase4_bench_auto"

    python_result = _run_mode(
        mode="python_only",
        language=language,
        sequences=sequences,
        latin_structural_ref=latin_structural_ref,
        latin_form_ref=latin_form_ref,
        references=references,
        family_reference_inventory=family_reference_inventory,
        output_dir=python_dir,
    )
    auto_result = _run_mode(
        mode="auto_batch",
        language=language,
        sequences=sequences,
        latin_structural_ref=latin_structural_ref,
        latin_form_ref=latin_form_ref,
        references=references,
        family_reference_inventory=family_reference_inventory,
        output_dir=auto_dir,
    )

    speedup = (
        python_result.seconds / auto_result.seconds
        if auto_result.seconds > 0.0
        else float("inf")
    )
    return {
        "language": language,
        "python_only": python_result.to_dict(),
        "auto_batch": auto_result.to_dict(),
        "speedup_auto_vs_python": round(speedup, 6),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark integrated v4 incremental guidance.")
    parser.add_argument("--language", default="french")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "validation" / "fortran_v4_phase4_benchmark.json",
    )
    args = parser.parse_args()

    result = run_benchmark(language=args.language)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
