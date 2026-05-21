"""
V5 Seed Replication Probe
=========================

Run a lightweight multi-seed replication under the same French -> Latin v5
condition family used for the paper run, but with a bounded proposal budget.

This is meant to harden the manuscript against the "single-run story" critique
without committing to another full plateau-scale production run.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.retrodiction.long_run_v4 import LongRunConfig
from src.retrodiction.long_run_v5 import run_long_continuation

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VALIDATION_DIR = PROJECT_ROOT / "data" / "validation"
DEFAULT_START_CORPUS = PROJECT_ROOT / "data" / "processed" / "romance" / "french_tokens.json"
PAPER_RUN_ROOT = PROJECT_ROOT / "data" / "retrodiction" / "french" / "v5_fortran_c16_seed45_paper_run"
PAPER_RUN_BLOCK1 = PAPER_RUN_ROOT / "blocks" / "block_0001" / "run_summary.json"


@dataclass(frozen=True)
class SeedResult:
    seed: int
    status: str
    block_count: int
    cumulative_proposals: int
    final_total_score: float | None
    final_latin_structural_score: float | None
    final_latin_form_score: float | None
    final_family_alignment_score: float | None
    accepted_mutation_stages: int | None
    final_coherence_label: str | None
    manifest_path: str
    summary_path: str
    final_corpus: str
    seed_audit: dict[str, Any] | None
    matches_historical_paper_block1: bool | None = None


def _probe_config(
    *,
    seed: int,
    output_dir: Path,
    start_corpus: Path,
    total_target_proposals: int,
    block_proposals: int,
    n_candidates: int,
    num_sequences: int,
    live_event_mode: str,
    live_event_buffer_size: int,
) -> LongRunConfig:
    return LongRunConfig(
        language="french",
        start_corpus=start_corpus,
        output_dir=output_dir,
        total_target_proposals=total_target_proposals,
        block_proposals=block_proposals,
        starting_proposals=0,
        num_sequences=num_sequences,
        n_candidates=n_candidates,
        max_accepted_stages=512,
        seed=seed,
        min_improvement=0.0001,
        struct_target=0.0,
        form_target=1.0,
        family_target=1.0,
        use_fortran_cosine=True,
        use_fortran_batch=True,
        use_incremental_scoring=True,
        use_semantic_transparency=False,
        transparency_weight=0.0,
        enable_culture_bombs=False,
        validator_set=[],
        validator_snapshot_every_blocks=0,
        live_event_mode=live_event_mode,
        live_event_buffer_size=live_event_buffer_size,
        plateau_window_blocks=10,
        plateau_struct_epsilon=0.001,
        plateau_form_epsilon=0.001,
        plateau_family_epsilon=0.001,
    )


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _historical_block1_match(summary: dict[str, Any]) -> bool | None:
    if not PAPER_RUN_BLOCK1.exists():
        return None
    historical = _load_json(PAPER_RUN_BLOCK1)
    keys = (
        "final_total_score",
        "final_latin_structural_score",
        "final_latin_form_score",
        "final_family_alignment_score",
        "accepted_mutation_stages",
        "best_stage_id",
    )
    return all(summary.get(key) == historical.get(key) for key in keys)


def _summarize_seed(manifest_path: Path) -> SeedResult:
    manifest = _load_json(manifest_path)
    blocks = manifest.get("blocks", [])
    if not blocks:
        raise RuntimeError(f"No blocks recorded in manifest: {manifest_path}")
    last_block = blocks[-1]
    summary_path = Path(last_block["summary_path"])
    summary = _load_json(summary_path)
    requested_seed = manifest.get("config", {}).get("seed")
    engine_seed = summary.get("seed_audit", {}).get("engine_seed")
    seed = int(engine_seed if engine_seed is not None else requested_seed)
    return SeedResult(
        seed=seed,
        status=str(manifest.get("status")),
        block_count=len(blocks),
        cumulative_proposals=int(manifest.get("cumulative_proposals", 0)),
        final_total_score=summary.get("final_total_score"),
        final_latin_structural_score=summary.get("final_latin_structural_score"),
        final_latin_form_score=summary.get("final_latin_form_score"),
        final_family_alignment_score=summary.get("final_family_alignment_score"),
        accepted_mutation_stages=summary.get("accepted_mutation_stages"),
        final_coherence_label=summary.get("final_coherence_label"),
        manifest_path=str(manifest_path),
        summary_path=str(summary_path),
        final_corpus=str(manifest.get("current_corpus")),
        seed_audit=summary.get("seed_audit"),
        matches_historical_paper_block1=(
            _historical_block1_match(summary) if seed == 42 else None
        ),
    )


def _result_dict(result: SeedResult) -> dict[str, Any]:
    return {
        "seed": result.seed,
        "status": result.status,
        "block_count": result.block_count,
        "cumulative_proposals": result.cumulative_proposals,
        "final_total_score": result.final_total_score,
        "final_latin_structural_score": result.final_latin_structural_score,
        "final_latin_form_score": result.final_latin_form_score,
        "final_family_alignment_score": result.final_family_alignment_score,
        "accepted_mutation_stages": result.accepted_mutation_stages,
        "final_coherence_label": result.final_coherence_label,
        "manifest_path": result.manifest_path,
        "summary_path": result.summary_path,
        "final_corpus": result.final_corpus,
        "seed_audit": result.seed_audit,
        "matches_historical_paper_block1": result.matches_historical_paper_block1,
    }


def run_probe(
    *,
    seeds: list[int],
    start_corpus: Path = DEFAULT_START_CORPUS,
    total_target_proposals: int = 1000,
    block_proposals: int = 1000,
    n_candidates: int = 16,
    num_sequences: int = 800,
    live_event_mode: str = "all",
    live_event_buffer_size: int = 64,
    output_root: Path | None = None,
) -> dict[str, Any]:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = output_root or (VALIDATION_DIR / f"french_v5_seed_replication_probe_{stamp}")
    root.mkdir(parents=True, exist_ok=True)

    results: list[SeedResult] = []
    for seed in seeds:
        seed_root = root / f"seed_{seed}"
        shutil.rmtree(seed_root, ignore_errors=True)
        cfg = _probe_config(
            seed=seed,
            output_dir=seed_root,
            start_corpus=start_corpus,
            total_target_proposals=total_target_proposals,
            block_proposals=block_proposals,
            n_candidates=n_candidates,
            num_sequences=num_sequences,
            live_event_mode=live_event_mode,
            live_event_buffer_size=live_event_buffer_size,
        )
        run_long_continuation(cfg)
        results.append(_summarize_seed(seed_root / "manifest.json"))

    structural_values = [r.final_latin_structural_score for r in results if r.final_latin_structural_score is not None]
    form_values = [r.final_latin_form_score for r in results if r.final_latin_form_score is not None]
    alignment_values = [r.final_family_alignment_score for r in results if r.final_family_alignment_score is not None]
    accepted_values = [r.accepted_mutation_stages for r in results if r.accepted_mutation_stages is not None]

    summary = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "start_corpus": str(start_corpus),
        "seeds": list(seeds),
        "total_target_proposals": total_target_proposals,
        "block_proposals": block_proposals,
        "n_candidates": n_candidates,
        "num_sequences": num_sequences,
        "use_incremental_scoring": True,
        "use_fortran_cosine": True,
        "use_fortran_batch": True,
        "use_semantic_transparency": False,
        "enable_culture_bombs": False,
        "live_event_mode": live_event_mode,
        "results": [_result_dict(result) for result in results],
        "ranges": {
            "final_latin_structural_score": {
                "min": min(structural_values) if structural_values else None,
                "max": max(structural_values) if structural_values else None,
            },
            "final_latin_form_score": {
                "min": min(form_values) if form_values else None,
                "max": max(form_values) if form_values else None,
            },
            "final_family_alignment_score": {
                "min": min(alignment_values) if alignment_values else None,
                "max": max(alignment_values) if alignment_values else None,
            },
            "accepted_mutation_stages": {
                "min": min(accepted_values) if accepted_values else None,
                "max": max(accepted_values) if accepted_values else None,
            },
        },
    }

    output_path = root / "summary.json"
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    return {
        "output_root": str(root),
        "summary_path": str(output_path),
        "summary": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a bounded multi-seed v5 replication probe for French -> Latin.")
    parser.add_argument("--start-corpus", type=Path, default=DEFAULT_START_CORPUS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 45])
    parser.add_argument("--total-target-proposals", type=int, default=1000)
    parser.add_argument("--block-proposals", type=int, default=1000)
    parser.add_argument("--n-candidates", type=int, default=16)
    parser.add_argument("--num-sequences", type=int, default=800)
    parser.add_argument("--live-event-mode", choices=["all", "selected", "accepted_only", "off"], default="all")
    parser.add_argument("--live-event-buffer-size", type=int, default=64)
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()

    result = run_probe(
        seeds=args.seeds,
        start_corpus=args.start_corpus,
        total_target_proposals=args.total_target_proposals,
        block_proposals=args.block_proposals,
        n_candidates=args.n_candidates,
        num_sequences=args.num_sequences,
        live_event_mode=args.live_event_mode,
        live_event_buffer_size=args.live_event_buffer_size,
        output_root=args.output_root,
    )
    print(result["summary_path"])


if __name__ == "__main__":
    main()
