"""
Long-Run V5 Continuation Driver
===============================
Purpose:
    Run a resumable long v5 continuation in local blocks.

    This mirrors the manifest/block machinery from long_run_v4, but builds
    ReinforcedV5Config objects and launches the actual v5 engine class.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from src.retrodiction.engine_reinforced_v5 import (
    ReinforcedV5Config,
    RelationalReinforcedRetrodictionEngineV5,
)
from src.retrodiction.long_run_v4 import LongRunConfig, run_long_continuation as _run_long_continuation

log = logging.getLogger(__name__)


def _default_v5_config(cfg: LongRunConfig, max_proposals: int) -> ReinforcedV5Config:
    return ReinforcedV5Config(
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
        enable_culture_bombs=cfg.enable_culture_bombs,
        live_event_mode=cfg.live_event_mode,
        live_event_buffer_size=cfg.live_event_buffer_size,
    )


def run_long_continuation(cfg: LongRunConfig) -> dict:
    return _run_long_continuation(
        cfg,
        engine_class=RelationalReinforcedRetrodictionEngineV5,
        config_builder=_default_v5_config,
    )


def _parse_args() -> LongRunConfig:
    parser = argparse.ArgumentParser(description="Run a resumable long v5 continuation in local blocks")
    parser.add_argument("--language", default="french")
    parser.add_argument("--start-corpus", type=Path, required=True, help="Path to the starting *_tokens.json corpus")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory where the block run will live")
    parser.add_argument("--target-proposals", type=int, default=1500, help="Total cumulative proposal budget")
    parser.add_argument("--block-proposals", type=int, default=1000, help="Proposal budget per local block")
    parser.add_argument("--starting-proposals", type=int, default=0, help="Already-consumed proposal budget before this run")
    parser.add_argument("--num-sequences", type=int, default=800)
    parser.add_argument("--n-candidates", type=int, default=32)
    parser.add_argument("--max-accepted-stages", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-improvement", type=float, default=0.0001)
    parser.add_argument("--struct-target", type=float, default=0.0)
    parser.add_argument("--form-target", type=float, default=1.0)
    parser.add_argument("--family-target", type=float, default=1.0)
    parser.add_argument("--use-fortran-cosine", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-fortran-batch", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-semantic-transparency", action="store_true")
    parser.add_argument("--transparency-weight", type=float, default=0.0)
    parser.add_argument("--enable-culture-bombs", action="store_true")
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
        use_fortran_cosine=args.use_fortran_cosine,
        use_fortran_batch=args.use_fortran_batch,
        use_semantic_transparency=args.use_semantic_transparency,
        transparency_weight=args.transparency_weight,
        enable_culture_bombs=args.enable_culture_bombs,
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
