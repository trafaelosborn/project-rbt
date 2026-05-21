"""
V5 Minimal CLI
==============
Usage:
    python -m src.control.cli <command> [options]

Commands:
    retrodact   Launch a new retrodiction run
    chain       Chain a new run from an existing manifest endpoint
    benchmark   Run a candidate-count throughput benchmark
    status      Show current status of a run
    validate    Score a run's best endpoint against the validator bank

This is an intentionally thin wrapper over RunController. Heavy UX work
belongs in the TUI, not here. If you need more options, use the TUI.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from src.control.run_config import (
    BLOCK_PROPOSALS_PRESETS,
    CANDIDATE_PRESETS,
    BenchmarkConfig,
    LIVE_EVENT_MODES,
    RECOMMENDED_V5_CANDIDATE_COUNT,
    RECOMMENDED_V5_USE_FORTRAN_BATCH,
    RunConfig,
)
from src.control.run_controller import RunController

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Shared argument builder
# ---------------------------------------------------------------------------

def _add_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--start-corpus", type=Path, required=True,
                        help="Path to starting *_tokens.json corpus")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Directory where the run will live")
    parser.add_argument("--language", default="french",
                        help="Source language (default: french)")
    parser.add_argument("--until-mode", default="latin_hit",
                        choices=["latin_hit", "plateau", "budget"],
                        help="Stopping condition (default: latin_hit)")
    parser.add_argument("--target-proposals", type=int, default=0,
                        help="Total proposal budget (required when --until-mode=budget)")
    parser.add_argument("--candidate-count", type=int, default=RECOMMENDED_V5_CANDIDATE_COUNT,
                        choices=list(CANDIDATE_PRESETS),
                        help=(
                            "Candidates per proposal "
                            f"(default: {RECOMMENDED_V5_CANDIDATE_COUNT}, presets: {CANDIDATE_PRESETS})"
                        ))
    parser.add_argument("--block-proposals", type=int, default=1000,
                        choices=list(BLOCK_PROPOSALS_PRESETS),
                        help=f"Proposals per block (default: 1000, presets: {BLOCK_PROPOSALS_PRESETS})")
    parser.add_argument("--num-sequences", type=int, default=800)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--fortran-batch",
        action=argparse.BooleanOptionalAction,
        default=RECOMMENDED_V5_USE_FORTRAN_BATCH,
        help=(
            "Enable the Fortran-backed batch candidate scoring path "
            f"(default: {'on' if RECOMMENDED_V5_USE_FORTRAN_BATCH else 'off'})"
        ),
    )
    parser.add_argument("--semantic-transparency", action="store_true", default=False,
                        help="Enable semantic transparency constraint (experimental condition)")
    parser.add_argument("--transparency-weight", type=float, default=0.0,
                        help="Weight for transparency term (required when --semantic-transparency)")
    parser.add_argument(
        "--validator-set",
        nargs="*",
        default=[],
        help="Optional validator corpus ids to snapshot against at block boundaries",
    )
    parser.add_argument(
        "--validator-snapshot-every-blocks",
        type=int,
        default=0,
        help="If > 0, write validator-bank snapshots every N completed blocks",
    )
    parser.add_argument(
        "--live-event-mode",
        default="all",
        choices=list(LIVE_EVENT_MODES),
        help="Live-event stream verbosity for Logan's Run style telemetry",
    )
    parser.add_argument(
        "--live-event-buffer-size",
        type=int,
        default=64,
        help="Number of live events to buffer before writing to disk",
    )


def _config_from_args(args: argparse.Namespace) -> RunConfig:
    return RunConfig(
        source_language=args.language,
        target_language="latin",
        start_corpus=args.start_corpus,
        until_mode=args.until_mode,
        total_target_proposals=args.target_proposals,
        candidate_count=args.candidate_count,
        block_proposals=args.block_proposals,
        num_sequences=args.num_sequences,
        seed=args.seed,
        use_fortran_batch=args.fortran_batch,
        use_semantic_transparency=args.semantic_transparency,
        transparency_weight=args.transparency_weight,
        validator_set=list(args.validator_set),
        validator_snapshot_every_blocks=args.validator_snapshot_every_blocks,
        live_event_mode=args.live_event_mode,
        live_event_buffer_size=args.live_event_buffer_size,
    )


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_retrodact(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    errors = config.validate()
    if errors:
        for e in errors:
            log.error(e)
        return 1
    for w in config.warnings():
        log.warning(w)

    controller = RunController()
    log.info("Launching run -> %s", args.output_dir)
    status = controller.launch_run(config, args.output_dir)
    print(status.summary_line())
    return 0


def cmd_chain(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    # validate before overriding start_corpus (corpus check happens after chain resolves it)
    errors = [e for e in config.validate() if "start_corpus" not in e]
    if errors:
        for e in errors:
            log.error(e)
        return 1
    for w in config.warnings():
        log.warning(w)

    controller = RunController()
    log.info("Chaining from %s -> %s", args.from_manifest, args.output_dir)
    status = controller.chain_run(args.from_manifest, config, args.output_dir)
    print(status.summary_line())
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    controller = RunController()
    status = controller.status(args.output_dir)
    print(status.summary_line())
    if args.json:
        print(json.dumps({
            "status": status.status,
            "cumulative_proposals": status.cumulative_proposals,
            "blocks_completed": status.blocks_completed,
            "last_struct": status.last_struct,
            "last_form": status.last_form,
            "last_align": status.last_align,
            "last_updated_utc": status.last_updated_utc,
            "current_corpus": status.current_corpus,
        }, indent=2))
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    controller = RunController()
    try:
        result = controller.validate_run(args.output_dir, output_path=args.output)
        summary = result.get("summary", {})
        nearest_structural = (summary.get("nearest_structural_by_block") or [{}])[-1]
        nearest_form = (summary.get("nearest_form_by_block") or [{}])[-1]
        print(
            f"Final nearest structural validator: {nearest_structural.get('validator_corpus', 'N/A')} "
            f"distance={nearest_structural.get('validator_structural_distance', float('nan')):.4f}"
        )
        print(
            f"Final nearest form validator: {nearest_form.get('validator_corpus', 'N/A')} "
            f"form={nearest_form.get('validator_form_score', float('nan')):.4f}"
        )
        print(f"CSV -> {result.get('csv_path')}")
        print(f"JSON -> {result.get('json_path')}")
        print(f"Chronology -> {result.get('summary_path')}")
    except (FileNotFoundError, ValueError) as exc:
        log.error(str(exc))
        return 1
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    config = BenchmarkConfig(
        source_language=args.language,
        start_corpus=args.start_corpus,
        candidate_counts=list(CANDIDATE_PRESETS),
        proposals_per_trial=args.proposals_per_trial,
        num_sequences=args.num_sequences,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    controller = RunController()
    result = controller.benchmark(config)
    print(f"Benchmark complete. Results -> {result.output_dir}")
    best = result.best_candidate_count()
    if best is not None:
        print(f"Best throughput at candidate_count={best}")
    for trial in result.trials:
        print(
            f"  c={trial['candidate_count']:3d}: "
            f"{trial['proposals_per_hour']:,.0f} proposals/hour "
            f"({trial['proposals_attempted']} proposals in {trial['wall_seconds']:.1f}s)"
        )
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.control.cli",
        description="V5 retrodiction controller CLI",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # retrodact
    p_retrodact = subparsers.add_parser("retrodact", help="Launch a new retrodiction run")
    _add_run_args(p_retrodact)
    p_retrodact.set_defaults(func=cmd_retrodact)

    # chain
    p_chain = subparsers.add_parser("chain", help="Chain a run from an existing manifest endpoint")
    _add_run_args(p_chain)
    p_chain.add_argument("--from-manifest", type=Path, required=True,
                         help="Path to the manifest.json of the run to chain from")
    p_chain.set_defaults(func=cmd_chain)

    # status
    p_status = subparsers.add_parser("status", help="Show current run status")
    p_status.add_argument("--output-dir", type=Path, required=True)
    p_status.add_argument("--json", action="store_true", help="Output as JSON")
    p_status.set_defaults(func=cmd_status)

    # validate
    p_validate = subparsers.add_parser("validate", help="Score best endpoint against validator bank")
    p_validate.add_argument("--output-dir", type=Path, required=True)
    p_validate.add_argument("--output", type=Path, default=None,
                            help="Optional output directory or file-like prefix path for validator-bank artifacts")
    p_validate.set_defaults(func=cmd_validate)

    # benchmark
    p_bench = subparsers.add_parser("benchmark", help="Run candidate-count throughput benchmark")
    p_bench.add_argument("--start-corpus", type=Path, required=True)
    p_bench.add_argument("--output-dir", type=Path, required=True)
    p_bench.add_argument("--language", default="french")
    p_bench.add_argument("--proposals-per-trial", type=int, default=200)
    p_bench.add_argument("--num-sequences", type=int, default=800)
    p_bench.add_argument("--seed", type=int, default=42)
    p_bench.set_defaults(func=cmd_benchmark)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
