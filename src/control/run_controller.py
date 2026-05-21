"""
V5 Run Controller
=================
Purpose:
    One controller that CLI, TUI, and tests all call into.

    The controller owns the boundary between the user-facing interface
    (CLI / TUI) and the engine internals (long_run driver, engine classes).
    Engine-specific details do not leak upward.

Design notes
------------
- `launch_run` validates, locks the config, then invokes the long-run driver.
- `chain_run` reads the last known corpus from an existing manifest and
  launches a new block run starting from that corpus.
- `status` reads a manifest without starting anything — safe to call any time.
- `stop_run` is a soft stop: it writes a sentinel file that the driver polls
  at block boundaries. It does not kill the process.
- `validate_run` scores the best endpoint from a finished manifest against
  the attested validator bank.
- `benchmark` runs a micro-sweep across candidate counts for throughput
  measurement; results are written to output_dir.

Fortran batch note
------------------
When `config.use_fortran_batch=True` the controller passes the flag through to
the v5 long-run driver. The current engine uses the compiled batch scorer for
candidate form scoring and a specialized batched structural/coherence
selection path around it, while accepted-stage artifact generation remains in
the standard Python path.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from src.control.run_config import BenchmarkConfig, RunConfig

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Status / result types
# ---------------------------------------------------------------------------

@dataclass
class RunStatus:
    """Snapshot of a run read from its manifest."""
    output_dir: Path
    status: str                      # "running" | "complete" | "latin_hit" | "stopped_manual" | "failed" | "not_found"
    cumulative_proposals: int
    blocks_completed: int
    last_struct: float | None
    last_form: float | None
    last_align: float | None
    last_updated_utc: str | None
    current_corpus: str | None
    last_error: str | None = None

    @property
    def is_terminal(self) -> bool:
        return self.status in {"complete", "latin_hit", "stopped_manual", "failed", "not_found"}

    def summary_line(self) -> str:
        struct = f"{self.last_struct:.4f}" if self.last_struct is not None else "N/A"
        form = f"{self.last_form:.4f}" if self.last_form is not None else "N/A"
        return (
            f"[{self.status}] proposals={self.cumulative_proposals} "
            f"blocks={self.blocks_completed} struct={struct} form={form}"
        )


@dataclass
class BenchmarkResult:
    """Result of a candidate-count benchmark sweep."""
    output_dir: Path
    trials: list[dict[str, Any]]

    def best_candidate_count(self) -> int | None:
        if not self.trials:
            return None
        return max(self.trials, key=lambda t: t.get("proposals_per_hour", 0)).get("candidate_count")


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------

class RunController:
    """
    Boundary layer between user interfaces and the engine internals.

    All public methods are safe to call from both CLI and TUI.
    Progress callbacks (if provided) receive a RunStatus at each block
    boundary so the TUI can refresh without polling.
    """

    def __init__(
        self,
        *,
        progress_callback: Callable[[RunStatus], None] | None = None,
    ) -> None:
        self._progress_callback = progress_callback

    # ------------------------------------------------------------------
    # Launch
    # ------------------------------------------------------------------

    def launch_run(
        self,
        config: RunConfig,
        output_dir: Path,
    ) -> RunStatus:
        """
        Validate, lock, and launch a new retrodiction run.

        The config is locked before the driver is called. Any attempt to
        mutate config after this point raises ConfigLockedError.
        """
        config.validate_strict()
        for warn in config.warnings():
            log.warning(warn)
        config.lock()

        from src.control._driver_adapter import run_driver
        run_driver(config, output_dir, progress_callback=self._progress_callback)
        return self.status(output_dir)

    def chain_run(
        self,
        from_manifest: Path,
        config: RunConfig,
        output_dir: Path,
    ) -> RunStatus:
        """
        Launch a new run chained from the best endpoint of an existing manifest.

        The start_corpus field in `config` is overridden with the best corpus
        from `from_manifest`. The config is then locked and launched.
        """
        if not from_manifest.exists():
            raise FileNotFoundError(f"Manifest not found: {from_manifest}")

        with from_manifest.open(encoding="utf-8") as fh:
            manifest = json.load(fh)

        current_corpus = manifest.get("current_corpus")
        if not current_corpus:
            raise ValueError(f"No current_corpus in manifest: {from_manifest}")

        # Override start_corpus with the chaining endpoint
        config.start_corpus = Path(current_corpus)
        log.info("Chaining from %s", current_corpus)

        return self.launch_run(config, output_dir)

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def status(self, output_dir: Path) -> RunStatus:
        """Read and return the current status of a run (non-destructive)."""
        manifest_path = output_dir / "manifest.json"
        if not manifest_path.exists():
            return RunStatus(
                output_dir=output_dir,
                status="not_found",
                cumulative_proposals=0,
                blocks_completed=0,
                last_struct=None,
                last_form=None,
                last_align=None,
                last_updated_utc=None,
                current_corpus=None,
                last_error=None,
            )

        with manifest_path.open(encoding="utf-8") as fh:
            manifest = json.load(fh)

        blocks = manifest.get("blocks", [])
        last = blocks[-1] if blocks else {}
        return RunStatus(
            output_dir=output_dir,
            status=manifest.get("status", "unknown"),
            cumulative_proposals=int(manifest.get("cumulative_proposals", 0)),
            blocks_completed=len(blocks),
            last_struct=last.get("final_latin_structural_score"),
            last_form=last.get("final_latin_form_score"),
            last_align=last.get("final_family_alignment_score"),
            last_updated_utc=manifest.get("updated_utc"),
            current_corpus=manifest.get("current_corpus"),
            last_error=manifest.get("last_error"),
        )

    # ------------------------------------------------------------------
    # Stop
    # ------------------------------------------------------------------

    def stop_run(self, output_dir: Path) -> None:
        """
        Request a soft stop at the next block boundary.

        Writes a stop-sentinel file. The driver checks for this file
        at the start of each block and halts cleanly if found.
        Does not kill any running process.
        """
        sentinel = output_dir / "stop_requested"
        sentinel.parent.mkdir(parents=True, exist_ok=True)
        sentinel.touch()
        log.info("Stop sentinel written to %s", sentinel)

    # ------------------------------------------------------------------
    # Validate
    # ------------------------------------------------------------------

    def validate_run(
        self,
        output_dir: Path,
        output_path: Path | None = None,
    ) -> dict[str, Any]:
        """
        Score the best endpoint from a completed run against the validator bank.

        Returns the validator-bank comparison dict.

        If `output_path` is provided, it is interpreted as either:
        - a directory, in which case the default run-id prefix is used
        - a file-like path, in which case the parent directory is used and
          the filename stem becomes the shared prefix for the CSV/JSON outputs
        """
        manifest_path = output_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"No manifest at {manifest_path}")

        with manifest_path.open(encoding="utf-8") as fh:
            manifest = json.load(fh)

        blocks = manifest.get("blocks", [])
        if not blocks:
            raise ValueError("Manifest has no completed blocks.")

        compare_output_dir: Path | None = None
        output_prefix: str | None = None
        if output_path is not None:
            if output_path.suffix:
                compare_output_dir = output_path.parent
                output_prefix = output_path.stem
            else:
                compare_output_dir = output_path

        from src.validation.validator_bank_compare import compare_run_manifest_to_validator_bank

        result = compare_run_manifest_to_validator_bank(
            run_manifest_path=manifest_path,
            output_prefix=output_prefix,
            output_dir=compare_output_dir,
        )
        return result

    # ------------------------------------------------------------------
    # Benchmark
    # ------------------------------------------------------------------

    def benchmark(self, config: BenchmarkConfig) -> BenchmarkResult:
        """
        Micro-benchmark proposals/hour across the configured candidate counts.

        Each trial runs `config.proposals_per_trial` proposals and records
        wall-clock throughput. Results are written to `config.output_dir`.
        """
        from src.control._benchmark_runner import run_candidate_count_benchmark

        config.output_dir.mkdir(parents=True, exist_ok=True)
        trials = run_candidate_count_benchmark(config)
        result_path = config.output_dir / "candidate_count_benchmark.json"
        with result_path.open("w", encoding="utf-8") as fh:
            json.dump(
                {
                    "run_utc": datetime.now(timezone.utc).isoformat(),
                    "config": config.to_dict(),
                    "trials": trials,
                },
                fh,
                ensure_ascii=False,
                indent=2,
            )
        log.info("Benchmark results written to %s", result_path)
        return BenchmarkResult(output_dir=config.output_dir, trials=trials)
