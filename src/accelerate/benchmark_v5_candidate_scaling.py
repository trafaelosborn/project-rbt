"""
Benchmark: V5 candidate scaling, plain Python vs Fortran batch scoring.

Usage:
    python -m src.accelerate.benchmark_v5_candidate_scaling
    python -m src.accelerate.benchmark_v5_candidate_scaling --proposals 50 --candidate-counts 8 16 32

This benchmark is for the real production question:
    as we widen v5's proposal fan-out, does the Fortran batch path buy us
    enough throughput to justify keeping it on?

Both modes run the same plain v5 search:
    - incremental scoring on
    - culture bombs off
    - semantic transparency off

The only difference is candidate scoring backend:
    - plain_python: Python candidate scoring
    - fortran_batch: Fortran batch form scoring enabled
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import time
from pathlib import Path

from src.retrodiction.engine_reinforced import LatinReference
from src.retrodiction.engine_reinforced_v2 import LatinFormReference
from src.retrodiction.engine_reinforced_v5 import (
    ReinforcedV5Config,
    RelationalReinforcedRetrodictionEngineV5,
)
from src.retrodiction.similarity import ReferenceSet
from src.validation.hungarian_alignment import extract_family_inventory

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_CORPUS = PROJECT_ROOT / "data/processed/romance/french_tokens.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data/benchmarks/v5_candidate_scaling"
DEFAULT_CANDIDATE_COUNTS = (8, 16, 32)
DEFAULT_PROPOSALS = 50
DEFAULT_SEED = 77
DEFAULT_NUM_SEQUENCES = 800


def _load_sequences(path: Path) -> list[list[str]]:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)["sequences"]


def _load_latin_sample() -> list[list[str]]:
    latin_path = PROJECT_ROOT / "data/sequestered/latin/latin_tokens.json"
    with latin_path.open(encoding="utf-8") as fh:
        return json.load(fh)["sequences"][:72]


def _run_probe(
    *,
    label: str,
    use_fortran_batch: bool,
    candidate_count: int,
    proposals: int,
    seed: int,
    sequences: list[list[str]],
    latin_structural_ref: LatinReference,
    latin_form_ref: LatinFormReference,
    references: ReferenceSet,
    family_ref,
    output_dir: Path,
) -> dict:
    cfg = ReinforcedV5Config(
        num_sequences=DEFAULT_NUM_SEQUENCES,
        max_proposals=proposals,
        max_accepted_stages=512,
        patience=proposals,
        seed=seed,
        n_candidates=candidate_count,
        min_improvement=0.0001,
        save_dense_matrices=False,
        use_incremental_scoring=True,
        use_fortran_cosine=use_fortran_batch,
        use_fortran_batch=use_fortran_batch,
        use_semantic_transparency=False,
        enable_culture_bombs=False,
    )

    engine = RelationalReinforcedRetrodictionEngineV5(
        language="french",
        source_sequences=list(sequences),
        latin_structural_ref=latin_structural_ref,
        latin_form_ref=latin_form_ref,
        config=cfg,
        output_dir=output_dir,
        references=references,
        family_reference_inventory=family_ref,
    )

    t0 = time.perf_counter()
    engine.run()
    elapsed = time.perf_counter() - t0

    with (output_dir / "run_summary.json").open(encoding="utf-8") as fh:
        summary = json.load(fh)

    stages = summary.get("stages", [])
    seed_stage = stages[0] if stages else None
    seed_struct = seed_stage["latin_structural_score"] if seed_stage else None
    seed_form = seed_stage["latin_form_score"] if seed_stage else None
    final_struct = summary.get("final_latin_structural_score")
    final_form = summary.get("final_latin_form_score")

    struct_gain = None
    if seed_struct is not None and final_struct is not None:
        struct_gain = float(final_struct) - float(seed_struct)

    form_gain = None
    if seed_form is not None and final_form is not None:
        form_gain = float(final_form) - float(seed_form)

    proposals_per_hour = summary["proposals_attempted"] / elapsed * 3600.0
    return {
        "label": label,
        "candidate_count": candidate_count,
        "proposals_attempted": summary["proposals_attempted"],
        "accepted_stages": summary["accepted_mutation_stages"],
        "wall_seconds": round(elapsed, 3),
        "proposals_per_hour": round(proposals_per_hour, 1),
        "final_struct": final_struct,
        "final_form": final_form,
        "final_alignment": summary.get("final_family_alignment_score"),
        "struct_gain_from_seed": None if struct_gain is None else round(struct_gain, 6),
        "form_gain_from_seed": None if form_gain is None else round(form_gain, 6),
        "halt_reason": summary.get("halt_reason"),
        "run_dir": str(output_dir),
    }


def run_scaling_benchmark(
    *,
    candidate_counts: list[int],
    proposals: int,
    seed: int,
    source_corpus: Path,
    output_dir: Path,
) -> dict:
    if not source_corpus.exists():
        raise FileNotFoundError(f"Source corpus not found: {source_corpus}")

    sequences = _load_sequences(source_corpus)
    latin_structural_ref = LatinReference()
    latin_form_ref = LatinFormReference()
    references = ReferenceSet()
    family_ref = extract_family_inventory(
        "latin",
        _load_latin_sample(),
        ReinforcedV5Config().alignment_config,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []

    for candidate_count in candidate_counts:
        row_results: list[dict] = []
        for label, use_fortran_batch in [
            ("plain_python", False),
            ("fortran_batch", True),
        ]:
            probe_dir = output_dir / f"{label}_{candidate_count}"
            shutil.rmtree(probe_dir, ignore_errors=True)
            probe_dir.mkdir(parents=True, exist_ok=True)
            log.info(
                "Running %s probe: candidates=%d proposals=%d",
                label,
                candidate_count,
                proposals,
            )
            result = _run_probe(
                label=label,
                use_fortran_batch=use_fortran_batch,
                candidate_count=candidate_count,
                proposals=proposals,
                seed=seed,
                sequences=sequences,
                latin_structural_ref=latin_structural_ref,
                latin_form_ref=latin_form_ref,
                references=references,
                family_ref=family_ref,
                output_dir=probe_dir,
            )
            row_results.append(result)
            results.append(result)
            log.info(
                "  %-14s candidates=%3d throughput=%8.1f p/h accepted=%2d struct=%+.4f form=%.4f",
                label,
                candidate_count,
                result["proposals_per_hour"],
                result["accepted_stages"],
                result["final_struct"] or 0.0,
                result["final_form"] or 0.0,
            )

        plain = next(item for item in row_results if item["label"] == "plain_python")
        fortran = next(item for item in row_results if item["label"] == "fortran_batch")
        speedup = fortran["proposals_per_hour"] / max(plain["proposals_per_hour"], 1e-9)
        log.info(
            "  speedup candidates=%3d: %.3fx (%0.1f -> %0.1f p/h)",
            candidate_count,
            speedup,
            plain["proposals_per_hour"],
            fortran["proposals_per_hour"],
        )

    grouped_rows = []
    for candidate_count in candidate_counts:
        plain = next(item for item in results if item["candidate_count"] == candidate_count and item["label"] == "plain_python")
        fortran = next(item for item in results if item["candidate_count"] == candidate_count and item["label"] == "fortran_batch")
        grouped_rows.append(
            {
                "candidate_count": candidate_count,
                "plain_python": plain,
                "fortran_batch": fortran,
                "speedup_fortran_vs_plain": round(
                    fortran["proposals_per_hour"] / max(plain["proposals_per_hour"], 1e-9),
                    6,
                ),
            }
        )

    report = {
        "source_corpus": str(source_corpus),
        "proposals": proposals,
        "seed": seed,
        "candidate_counts": list(candidate_counts),
        "results": grouped_rows,
    }
    out_path = output_dir / "scaling_report.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    log.info("Scaling report written to %s", out_path)
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-counts",
        nargs="+",
        type=int,
        default=list(DEFAULT_CANDIDATE_COUNTS),
        help="Candidate counts to benchmark (default: 8 16 32).",
    )
    parser.add_argument(
        "--proposals",
        type=int,
        default=DEFAULT_PROPOSALS,
        help=f"Proposals per probe (default: {DEFAULT_PROPOSALS}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"RNG seed for all probes (default: {DEFAULT_SEED}).",
    )
    parser.add_argument(
        "--source-corpus",
        type=Path,
        default=DEFAULT_SOURCE_CORPUS,
        help="Source corpus JSON to start from.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for benchmark artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = run_scaling_benchmark(
        candidate_counts=args.candidate_counts,
        proposals=args.proposals,
        seed=args.seed,
        source_corpus=args.source_corpus,
        output_dir=args.output_dir,
    )
    for row in report["results"]:
        plain = row["plain_python"]
        fortran = row["fortran_batch"]
        print(
            f"candidates={row['candidate_count']:>3}  "
            f"plain={plain['proposals_per_hour']:>8.1f} p/h  "
            f"fortran={fortran['proposals_per_hour']:>8.1f} p/h  "
            f"speedup={row['speedup_fortran_vs_plain']:.3f}x"
        )


if __name__ == "__main__":
    main()
