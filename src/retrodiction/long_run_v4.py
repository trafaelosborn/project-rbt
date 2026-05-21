"""
Long-Run V4 Continuation Driver
===============================
Purpose:
    Run a long v4 continuation in resumable local blocks.

    This keeps the heavy search entirely local while making multi-hour runs
    robust against shell/session timeouts.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.retrodiction.engine_reinforced import LatinReference
from src.retrodiction.engine_reinforced_v2 import LatinFormReference
from src.retrodiction.engine_reinforced_v4 import (
    ReinforcedV4Config,
    RelationalReinforcedRetrodictionEngineV4,
)
from src.retrodiction.similarity import ReferenceSet
from src.validation.validator_bank_compare import compare_run_manifest_to_validator_bank

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class LongRunConfig:
    """Top-level config for the resumable block driver."""

    language: str
    start_corpus: Path
    output_dir: Path
    total_target_proposals: int = 1500
    block_proposals: int = 200
    starting_proposals: int = 0
    num_sequences: int = 800
    n_candidates: int = 8
    max_accepted_stages: int = 512
    seed: int = 42
    min_improvement: float = 0.0001
    struct_target: float = 0.0
    form_target: float = 1.0
    family_target: float = 1.0
    use_fortran_cosine: bool = True          # Fortran/BLAS dense cosine in incremental scoring
    use_fortran_batch: bool = False          # Batch candidate form scoring in proposal loop
    use_incremental_scoring: bool = True
    use_semantic_transparency: bool = False
    transparency_weight: float = 0.0
    enable_culture_bombs: bool = False
    validator_set: list[str] = field(default_factory=list)
    validator_snapshot_every_blocks: int = 0
    live_event_mode: str = "all"
    live_event_buffer_size: int = 64
    plateau_window_blocks: int = 10
    plateau_struct_epsilon: float = 0.001
    plateau_form_epsilon: float = 0.001
    plateau_family_epsilon: float = 0.001

    def to_dict(self) -> dict[str, Any]:
        return {
            "language": self.language,
            "start_corpus": str(self.start_corpus),
            "output_dir": str(self.output_dir),
            "total_target_proposals": self.total_target_proposals,
            "block_proposals": self.block_proposals,
            "starting_proposals": self.starting_proposals,
            "num_sequences": self.num_sequences,
            "n_candidates": self.n_candidates,
            "max_accepted_stages": self.max_accepted_stages,
            "seed": self.seed,
            "min_improvement": self.min_improvement,
            "struct_target": self.struct_target,
            "form_target": self.form_target,
            "family_target": self.family_target,
            "use_fortran_cosine": self.use_fortran_cosine,
            "use_fortran_batch": self.use_fortran_batch,
            "use_incremental_scoring": self.use_incremental_scoring,
            "use_semantic_transparency": self.use_semantic_transparency,
            "transparency_weight": self.transparency_weight,
            "enable_culture_bombs": self.enable_culture_bombs,
            "validator_set": list(self.validator_set),
            "validator_snapshot_every_blocks": self.validator_snapshot_every_blocks,
            "live_event_mode": self.live_event_mode,
            "live_event_buffer_size": self.live_event_buffer_size,
            "plateau_window_blocks": self.plateau_window_blocks,
            "plateau_struct_epsilon": self.plateau_struct_epsilon,
            "plateau_form_epsilon": self.plateau_form_epsilon,
            "plateau_family_epsilon": self.plateau_family_epsilon,
        }


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_sequences(corpus_path: Path) -> list[list[str]]:
    with corpus_path.open(encoding="utf-8") as fh:
        return json.load(fh)["sequences"]


def _load_manifest(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _save_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)


def _seed_audit_record(requested_seed: int | None, engine_seed: int | None) -> dict[str, Any]:
    return {
        "requested_seed": requested_seed,
        "engine_seed": engine_seed,
        "seeds_match": (
            None
            if requested_seed is None or engine_seed is None
            else requested_seed == engine_seed
        ),
    }


def _apply_seed_audit_to_summary(
    summary_path: Path,
    summary: dict[str, Any],
    *,
    requested_seed: int | None,
    engine_seed: int | None,
) -> dict[str, Any]:
    """
    Normalize seed metadata in a block summary and stamp an explicit seed audit.

    This keeps the run summary aligned with the actual engine configuration even
    if a stale/default seed slips into downstream serialization.
    """
    config = summary.get("config")
    if not isinstance(config, dict):
        config = {}
        summary["config"] = config

    if engine_seed is not None:
        config["seed"] = engine_seed
    if requested_seed is not None:
        config["requested_seed"] = requested_seed

    summary["seed_audit"] = _seed_audit_record(requested_seed, engine_seed)

    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    return summary


def _remaining_proposals(total_target: int, starting: int, completed: int) -> int:
    if total_target <= 0:
        return 10**12
    return max(total_target - starting - completed, 0)


def _next_block_size(total_target: int, block_size: int, starting: int, completed: int) -> int:
    return min(block_size, _remaining_proposals(total_target, starting, completed))


def _joint_hit(
    summary: dict[str, Any],
    struct_target: float,
    form_target: float,
    family_target: float,
) -> bool:
    struct_raw = summary.get("final_latin_structural_score")
    form_raw = summary.get("final_latin_form_score")
    family_raw = summary.get("final_family_alignment_score")
    struct_score = float("-inf") if struct_raw is None else float(struct_raw)
    form_score = float("-inf") if form_raw is None else float(form_raw)
    family_score = float("-inf") if family_raw is None else float(family_raw)
    return (
        struct_score >= struct_target
        and form_score >= form_target
        and family_score >= family_target
    )


def _best_metric(blocks: list[dict[str, Any]], key: str) -> float:
    values = [block.get(key) for block in blocks]
    numeric = [float(value) for value in values if value is not None]
    return max(numeric) if numeric else float("-inf")


def _joint_plateau_hit(manifest: dict[str, Any], cfg: LongRunConfig) -> bool:
    blocks = manifest.get("blocks", [])
    window = cfg.plateau_window_blocks
    if window <= 0 or len(blocks) < window:
        return False

    recent = blocks[-window:]
    prior = blocks[:-window]
    if not prior:
        return False

    prior_struct = _best_metric(prior, "final_latin_structural_score")
    prior_form = _best_metric(prior, "final_latin_form_score")
    prior_family = _best_metric(prior, "final_family_alignment_score")

    recent_struct = _best_metric(recent, "final_latin_structural_score")
    recent_form = _best_metric(recent, "final_latin_form_score")
    recent_family = _best_metric(recent, "final_family_alignment_score")

    struct_improved = recent_struct > prior_struct + cfg.plateau_struct_epsilon
    form_improved = recent_form > prior_form + cfg.plateau_form_epsilon
    family_improved = recent_family > prior_family + cfg.plateau_family_epsilon

    return not (struct_improved or form_improved or family_improved)


def _maybe_write_validator_snapshot(
    manifest_path: Path,
    cfg: LongRunConfig,
    block_name: str,
) -> dict[str, Any] | None:
    every = int(cfg.validator_snapshot_every_blocks)
    if every <= 0:
        return None

    block_num = int(block_name.split("_")[-1])
    if block_num % every != 0:
        return None

    snapshot_dir = cfg.output_dir / "validator_snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    output_prefix = f"{cfg.output_dir.name}_{block_name}"
    result = compare_run_manifest_to_validator_bank(
        run_manifest_path=manifest_path,
        output_prefix=output_prefix,
        validator_ids=cfg.validator_set or None,
        output_dir=snapshot_dir,
        block_ids=[block_name],
    )
    return {
        "csv_path": result["csv_path"],
        "json_path": result["json_path"],
        "summary_path": result["summary_path"],
        "validator_count": result.get("validator_count"),
        "block_count": result.get("block_count"),
    }


def _default_v4_config(cfg: LongRunConfig, max_proposals: int) -> ReinforcedV4Config:
    return ReinforcedV4Config(
        num_sequences=cfg.num_sequences,
        max_proposals=max_proposals,
        max_accepted_stages=cfg.max_accepted_stages,
        patience=max_proposals,
        seed=cfg.seed,
        n_candidates=cfg.n_candidates,
        min_improvement=cfg.min_improvement,
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
        save_dense_matrices=False,
        use_incremental_scoring=cfg.use_incremental_scoring,
        use_fortran_cosine=cfg.use_fortran_cosine,
        use_fortran_batch=cfg.use_fortran_batch,
        use_semantic_transparency=cfg.use_semantic_transparency,
        transparency_weight=cfg.transparency_weight,
        live_event_mode=cfg.live_event_mode,
        live_event_buffer_size=cfg.live_event_buffer_size,
    )


def _initial_manifest(cfg: LongRunConfig) -> dict[str, Any]:
    return {
        "created_utc": _utc_now_iso(),
        "updated_utc": _utc_now_iso(),
        "status": "running",
        "last_error": None,
        "config": cfg.to_dict(),
        "seed_audit": _seed_audit_record(cfg.seed, None),
        "starting_proposals": cfg.starting_proposals,
        "completed_block_proposals": 0,
        "cumulative_proposals": cfg.starting_proposals,
        "current_corpus": str(cfg.start_corpus),
        "latin_hit": False,
        "joint_hit": False,
        "plateau_hit": False,
        "blocks": [],
    }


def run_long_continuation(
    cfg: LongRunConfig,
    engine_class=RelationalReinforcedRetrodictionEngineV4,
    config_builder=None,
) -> dict[str, Any]:
    manifest_path = cfg.output_dir / "manifest.json"
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    (cfg.output_dir / "blocks").mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest(manifest_path) or _initial_manifest(cfg)

    if manifest.get("status") in {"complete", "joint_hit", "plateau_hit"}:
        return manifest

    manifest["status"] = "running"
    manifest["last_error"] = None
    manifest["updated_utc"] = _utc_now_iso()
    _save_manifest(manifest_path, manifest)

    block_index = len(manifest["blocks"])
    completed = int(manifest.get("completed_block_proposals", 0))
    current_corpus = Path(manifest.get("current_corpus", cfg.start_corpus))

    latin_structural_ref = LatinReference()
    latin_form_ref = LatinFormReference()
    references = ReferenceSet()
    config_builder = config_builder or _default_v4_config

    try:
        while True:
            next_size = _next_block_size(
                cfg.total_target_proposals,
                cfg.block_proposals,
                cfg.starting_proposals,
                completed,
            )
            if next_size <= 0:
                manifest["status"] = "complete"
                break

            block_name = f"block_{block_index + 1:04d}"
            block_dir = cfg.output_dir / "blocks" / block_name
            log.info(
                "Starting %s from %s with %d proposals (completed=%d / target=%d)",
                block_name,
                current_corpus,
                next_size,
                cfg.starting_proposals + completed,
                cfg.total_target_proposals if cfg.total_target_proposals > 0 else -1,
            )

            block_input_corpus = current_corpus
            sequences = _load_sequences(block_input_corpus)
            engine_cfg = config_builder(cfg, max_proposals=next_size)
            engine = engine_class(
                language=cfg.language,
                source_sequences=sequences,
                latin_structural_ref=latin_structural_ref,
                latin_form_ref=latin_form_ref,
                config=engine_cfg,
                output_dir=block_dir,
                references=references,
            )
            engine.run()

            summary_path = block_dir / "run_summary.json"
            with summary_path.open(encoding="utf-8") as fh:
                summary = json.load(fh)
            summary = _apply_seed_audit_to_summary(
                summary_path,
                summary,
                requested_seed=cfg.seed,
                engine_seed=getattr(engine_cfg, "seed", None),
            )

            block_proposals_used = int(summary.get("proposals_attempted", next_size))
            completed += block_proposals_used
            current_corpus = Path(summary["best_corpus_json"])
            manifest["current_corpus"] = str(current_corpus)
            manifest["completed_block_proposals"] = completed
            manifest["cumulative_proposals"] = cfg.starting_proposals + completed
            manifest["seed_audit"] = _seed_audit_record(cfg.seed, getattr(engine_cfg, "seed", None))
            manifest["blocks"].append(
                {
                    "block": block_name,
                    "started_from_corpus": str(block_input_corpus),
                    "summary_path": str(summary_path),
                    "proposals_attempted": summary.get("proposals_attempted"),
                    "accepted_mutation_stages": summary.get("accepted_mutation_stages"),
                    "halt_reason": summary.get("halt_reason"),
                    "best_stage_id": summary.get("best_stage_id"),
                    "best_corpus_json": summary.get("best_corpus_json"),
                    "best_preview_txt": summary.get("best_preview_txt"),
                    "final_latin_structural_score": summary.get("final_latin_structural_score"),
                    "final_latin_form_score": summary.get("final_latin_form_score"),
                    "final_family_alignment_score": summary.get("final_family_alignment_score"),
                    "final_coherence_label": summary.get("final_coherence_label"),
                    "seed_audit": summary.get("seed_audit"),
                    "ended_at_corpus": str(current_corpus),
                }
            )

            validator_snapshot = _maybe_write_validator_snapshot(manifest_path, cfg, block_name)
            if validator_snapshot is not None:
                manifest["blocks"][-1]["validator_snapshot"] = validator_snapshot

            manifest["updated_utc"] = _utc_now_iso()
            manifest["joint_hit"] = _joint_hit(
                summary,
                cfg.struct_target,
                cfg.form_target,
                cfg.family_target,
            )
            manifest["latin_hit"] = manifest["joint_hit"]
            manifest["plateau_hit"] = _joint_plateau_hit(manifest, cfg)
            _save_manifest(manifest_path, manifest)

            if manifest["joint_hit"]:
                manifest["status"] = "joint_hit"
                break

            if manifest["plateau_hit"]:
                manifest["status"] = "plateau_hit"
                break

            block_index += 1
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["last_error"] = f"{type(exc).__name__}: {exc}"
        manifest["updated_utc"] = _utc_now_iso()
        _save_manifest(manifest_path, manifest)
        raise

    manifest["updated_utc"] = _utc_now_iso()
    _save_manifest(manifest_path, manifest)
    return manifest


def _parse_args() -> LongRunConfig:
    parser = argparse.ArgumentParser(description="Run a resumable long v4 continuation in local blocks")
    parser.add_argument("--language", default="french")
    parser.add_argument("--start-corpus", type=Path, required=True, help="Path to the starting *_tokens.json corpus")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory where the block run will live")
    parser.add_argument("--target-proposals", type=int, default=1500, help="Total cumulative proposal budget")
    parser.add_argument("--block-proposals", type=int, default=200, help="Proposal budget per local block")
    parser.add_argument("--starting-proposals", type=int, default=0, help="Already-consumed proposal budget before this run")
    parser.add_argument("--num-sequences", type=int, default=800)
    parser.add_argument("--n-candidates", type=int, default=8)
    parser.add_argument("--max-accepted-stages", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-improvement", type=float, default=0.0001)
    parser.add_argument("--struct-target", type=float, default=0.0)
    parser.add_argument("--form-target", type=float, default=1.0)
    parser.add_argument("--family-target", type=float, default=1.0)
    parser.add_argument("--validator-set", nargs="*", default=[])
    parser.add_argument("--validator-snapshot-every-blocks", type=int, default=0)
    parser.add_argument("--live-event-mode", default="all", choices=["all", "selected", "accepted_only", "off"])
    parser.add_argument("--live-event-buffer-size", type=int, default=64)
    parser.add_argument("--plateau-window-blocks", type=int, default=10)
    parser.add_argument("--plateau-struct-epsilon", type=float, default=0.001)
    parser.add_argument("--plateau-form-epsilon", type=float, default=0.001)
    parser.add_argument("--plateau-family-epsilon", type=float, default=0.001)
    args = parser.parse_args()
    return LongRunConfig(
        language=args.language,
        start_corpus=args.start_corpus,
        output_dir=args.output_dir,
        total_target_proposals=args.target_proposals,
        block_proposals=args.block_proposals,
        starting_proposals=args.starting_proposals,
        num_sequences=args.num_sequences,
        n_candidates=args.n_candidates,
        max_accepted_stages=args.max_accepted_stages,
        seed=args.seed,
        min_improvement=args.min_improvement,
        struct_target=args.struct_target,
        form_target=args.form_target,
        family_target=args.family_target,
        validator_set=list(args.validator_set),
        validator_snapshot_every_blocks=args.validator_snapshot_every_blocks,
        live_event_mode=args.live_event_mode,
        live_event_buffer_size=args.live_event_buffer_size,
        plateau_window_blocks=args.plateau_window_blocks,
        plateau_struct_epsilon=args.plateau_struct_epsilon,
        plateau_form_epsilon=args.plateau_form_epsilon,
        plateau_family_epsilon=args.plateau_family_epsilon,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = _parse_args()
    manifest = run_long_continuation(cfg)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
