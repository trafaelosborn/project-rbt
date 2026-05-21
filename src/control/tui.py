"""
V5 TUI Scaffold
===============
Usage:
    python -m src.control.tui

Requires: textual (pip install textual)

Architecture
------------
The TUI is a thin skin over RunController. All config, launch, and status
logic lives in run_controller.py. The TUI adds:
  - interactive launch-time config panels
  - live score/proposal-rate display during a run
  - config lock enforcement (fields become read-only once the run starts)

The TUI does NOT allow methodology changes after launch. Every toggle
(fortran_batch, semantic_transparency, candidate_count, block_proposals)
is a launch-time option only.

Panels
------
- ConfigPanel       launch-time toggles and corpus selection
- RunStatusPanel    live struct / form / alignment scores
- ThroughputPanel   wall clock, pace, and in-run progress
- LogansRunPanel    live comparison versus frozen v4 baseline
- LiveMovePanel     recent candidate/move stream
- ValidatorBankPanel latest attested-validator comparison summary
- LogPanel          INFO-level engine log stream

Flow
----
  Launch screen:
    ConfigPanel → [Launch] → locks config, starts driver in thread
  Live screen:
    RunStatusPanel + ThroughputPanel + PreviewPanel + LogPanel
    [Stop] → writes stop sentinel at next block boundary
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.containers import Container, Horizontal, Vertical
    from textual.reactive import reactive
    from textual.screen import Screen
    from textual.widgets import (
        Button,
        Checkbox,
        Footer,
        Header,
        Input,
        Label,
        Log,
        Select,
        Static,
    )
    _TEXTUAL_AVAILABLE = True
except ImportError:
    _TEXTUAL_AVAILABLE = False

from src.control.run_config import (
    BLOCK_PROPOSALS_PRESETS,
    CANDIDATE_PRESETS,
    RECOMMENDED_V5_CANDIDATE_COUNT,
    RECOMMENDED_V5_USE_FORTRAN_BATCH,
    RunConfig,
)
from src.control.run_controller import RunController, RunStatus

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VALIDATION_DIR = PROJECT_ROOT / "data" / "validation"
V4_BASELINE_REPORT = PROJECT_ROOT / "data" / "benchmarks" / "v5_vs_v4_headtohead" / "headtohead_report.json"
V4_BASELINE_EVENTS = PROJECT_ROOT / "data" / "benchmarks" / "v5_vs_v4_headtohead" / "v4_baseline" / "live_events.jsonl"


@dataclass(frozen=True)
class _V4BaselineReference:
    proposals_per_hour: float
    aggregate_mean_score_delta: float
    operator_mean_score_delta: dict[str, float]


def _format_hms(seconds: float) -> str:
    total = max(0, int(seconds))
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def _relative_lift(current: float, baseline: float) -> float | None:
    if abs(baseline) < 1e-9:
        return None
    return (current - baseline) / abs(baseline) * 100.0


def _load_v4_baseline_reference() -> _V4BaselineReference | None:
    if not V4_BASELINE_REPORT.exists() or not V4_BASELINE_EVENTS.exists():
        return None

    try:
        report = json.loads(V4_BASELINE_REPORT.read_text(encoding="utf-8"))
        v4_result = next(
            result for result in report.get("results", [])
            if result.get("label") == "v4_baseline"
        )
        proposals_per_hour = float(v4_result["proposals_per_hour"])

        operator_deltas: dict[str, list[float]] = defaultdict(list)
        aggregate: list[float] = []
        for line in V4_BASELINE_EVENTS.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            delta = event.get("score_delta")
            operator = event.get("operator")
            if delta is None or not operator:
                continue
            value = float(delta)
            aggregate.append(value)
            operator_deltas[str(operator)].append(value)

        if not aggregate:
            return None

        return _V4BaselineReference(
            proposals_per_hour=proposals_per_hour,
            aggregate_mean_score_delta=sum(aggregate) / len(aggregate),
            operator_mean_score_delta={
                operator: sum(values) / len(values)
                for operator, values in operator_deltas.items()
                if values
            },
        )
    except Exception:
        log.debug("Could not load v4 baseline comparison artifacts.", exc_info=True)
        return None


def _find_latest_validator_summary(output_dir: Path) -> Path | None:
    """Find the newest validator-bank chronology artifact for a run."""
    candidates: list[Path] = []
    snapshot_dir = output_dir / "validator_snapshots"
    run_id = output_dir.name

    if snapshot_dir.exists():
        candidates.extend(snapshot_dir.glob("*_vs_validator_bank_chronology.json"))
        candidates.extend(snapshot_dir.glob("*_chronology.json"))

    if VALIDATION_DIR.exists():
        candidates.extend(VALIDATION_DIR.glob(f"*{run_id}*_vs_validator_bank_chronology.json"))
        candidates.extend(VALIDATION_DIR.glob(f"*{run_id}*_chronology.json"))

    seen: set[str] = set()
    existing: list[Path] = []
    for path in candidates:
        key = str(path)
        if key in seen or not path.exists():
            continue
        seen.add(key)
        existing.append(path)

    if not existing:
        return None

    try:
        return max(existing, key=lambda path: path.stat().st_mtime)
    except OSError:
        return existing[-1]


def _load_validator_summary(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        log.debug("Could not load validator summary from %s", path, exc_info=True)
        return None


def _validator_bank_lines(
    summary: dict[str, Any] | None,
    *,
    summary_path: Path | None,
    status_text: str | None = None,
) -> list[str]:
    lines: list[str] = []
    if status_text:
        lines.append(f"Status    : {status_text}")

    if summary is None:
        if summary_path is None:
            lines.append("Status    : no validator-bank artifacts yet")
            lines.append("Hint      : press [v] to run full-bank compare")
        else:
            lines.append(f"Source    : {summary_path.name}")
            lines.append("Status    : validator-bank summary could not be parsed")
        return lines

    lines.append(f"Source    : {(summary_path.name if summary_path else 'in-memory summary')}")
    validator_count = summary.get("validator_count")
    block_count = summary.get("block_count")
    if validator_count is not None and block_count is not None:
        lines.append(f"Coverage  : {validator_count} validators across {block_count} blocks")

    nearest_structural = summary.get("nearest_structural_by_block") or []
    nearest_form = summary.get("nearest_form_by_block") or []

    if nearest_structural:
        item = nearest_structural[-1]
        corpus = item.get("validator_corpus", "N/A")
        period = item.get("validator_period", "?")
        start = item.get("validator_date_start", "?")
        end = item.get("validator_date_end", "?")
        distance = item.get("validator_structural_distance")
        distance_text = "n/a" if distance is None else f"{float(distance):.4f}"
        lines.append(
            f"Struct    : {corpus} ({period}, {start}-{end}) dist={distance_text}"
        )

    if nearest_form:
        item = nearest_form[-1]
        corpus = item.get("validator_corpus", "N/A")
        period = item.get("validator_period", "?")
        start = item.get("validator_date_start", "?")
        end = item.get("validator_date_end", "?")
        score = item.get("validator_form_score")
        score_text = "n/a" if score is None else f"{float(score):.4f}"
        lines.append(
            f"Form      : {corpus} ({period}, {start}-{end}) form={score_text}"
        )

    structural_path = summary.get("structural_path") or []
    if structural_path:
        lines.append(f"S path    : {' -> '.join(structural_path[-5:])}")

    form_path = summary.get("form_path") or []
    if form_path:
        lines.append(f"F path    : {' -> '.join(form_path[-5:])}")

    return lines


class _LiveEventAccumulator:
    """Incrementally ingest live event files without re-reading old content."""

    def __init__(self, max_recent: int = 10) -> None:
        self._offsets: dict[Path, int] = {}
        self._recent_events: deque[dict[str, Any]] = deque(maxlen=max_recent)
        self._aggregate_sum = 0.0
        self._aggregate_count = 0
        self._operator_sum: dict[str, float] = defaultdict(float)
        self._operator_count: dict[str, int] = defaultdict(int)

    def ingest_run_dir(self, output_dir: Path) -> list[dict[str, Any]]:
        blocks_dir = output_dir / "blocks"
        if not blocks_dir.exists():
            return list(self._recent_events)

        for path in sorted(blocks_dir.glob("block_*/live_events.jsonl")):
            self._ingest_file(path)
        return list(self._recent_events)

    def aggregate_mean_score_delta(self) -> float | None:
        if self._aggregate_count <= 0:
            return None
        return self._aggregate_sum / self._aggregate_count

    def top_operator_means(self, limit: int = 4) -> list[tuple[str, float, int]]:
        rows = []
        for operator, count in self._operator_count.items():
            if count <= 0:
                continue
            rows.append((operator, self._operator_sum[operator] / count, count))
        rows.sort(key=lambda row: (-row[2], row[0]))
        return rows[:limit]

    def _ingest_file(self, path: Path) -> None:
        previous = self._offsets.get(path, 0)
        try:
            size = path.stat().st_size
            if size < previous:
                previous = 0

            with path.open("r", encoding="utf-8") as fh:
                fh.seek(previous)
                for line in fh:
                    if not line.strip():
                        continue
                    event = json.loads(line)
                    self._recent_events.append(event)

                    delta = event.get("score_delta")
                    operator = event.get("operator")
                    if delta is not None and operator:
                        value = float(delta)
                        key = str(operator)
                        self._aggregate_sum += value
                        self._aggregate_count += 1
                        self._operator_sum[key] += value
                        self._operator_count[key] += 1

                self._offsets[path] = fh.tell()
        except Exception:
            log.debug("Could not ingest live event file %s", path, exc_info=True)


# ---------------------------------------------------------------------------
# Textual-unavailable fallback
# ---------------------------------------------------------------------------

def _no_textual() -> None:
    print(
        "The TUI requires textual. Install it with:\n"
        "    pip install textual\n\n"
        "Or use the CLI:\n"
        "    python -m src.control.cli --help"
    )


# ---------------------------------------------------------------------------
# Launch screen
# ---------------------------------------------------------------------------

if _TEXTUAL_AVAILABLE:

    class ConfigPanel(Vertical):
        """Launch-time configuration panel. All fields lock after [Launch]."""

        DEFAULT_CSS = """
        ConfigPanel {
            border: solid $accent;
            padding: 1 2;
            height: auto;
        }
        ConfigPanel Label {
            margin-bottom: 1;
        }
        ConfigPanel Input {
            margin-bottom: 1;
        }
        """

        def compose(self) -> ComposeResult:
            yield Label("[b]Source corpus[/b]")
            yield Input(placeholder="/path/to/corpus_tokens.json", id="start_corpus")
            yield Label("[b]Output directory[/b]")
            yield Input(placeholder="/path/to/output_dir", id="output_dir")
            yield Label(f"[b]Candidate count[/b] ({RECOMMENDED_V5_CANDIDATE_COUNT} recommended)")
            yield Select(
                [(str(c), str(c)) for c in CANDIDATE_PRESETS],
                value=str(RECOMMENDED_V5_CANDIDATE_COUNT),
                id="candidate_count",
            )
            yield Label("[b]Block proposals[/b]")
            yield Select(
                [(str(b), str(b)) for b in BLOCK_PROPOSALS_PRESETS],
                value="1000",
                id="block_proposals",
            )
            yield Checkbox("Incremental scoring (recommended)", value=True, id="use_incremental_scoring")
            yield Checkbox(
                "Fortran batch scoring (recommended)",
                value=RECOMMENDED_V5_USE_FORTRAN_BATCH,
                id="use_fortran_batch",
            )
            yield Checkbox("Semantic transparency constraint (experimental)", value=False, id="use_semantic_transparency")
            yield Label("[b]Transparency weight[/b] (0 = disabled)")
            yield Input(placeholder="0.0", value="0.0", id="transparency_weight")
            yield Label("[b]Validator snapshots every N blocks[/b] (0 = off)")
            yield Input(placeholder="0", value="0", id="validator_snapshot_every_blocks")
            yield Label("[b]Validator set[/b] (optional, comma-separated corpus ids)")
            yield Input(placeholder="old_french,middle_french", value="", id="validator_set")
            yield Label("[b]Seed[/b]")
            yield Input(placeholder="42", value="42", id="seed")

        def build_config(self) -> RunConfig:
            """Read field values and return an unlocked RunConfig."""
            corpus = self.query_one("#start_corpus", Input).value.strip()
            output = self.query_one("#output_dir", Input).value.strip()
            if not corpus:
                raise ValueError("start_corpus is required")
            if not output:
                raise ValueError("output_dir is required")
            n_cand = int(self.query_one("#candidate_count", Select).value)
            n_block = int(self.query_one("#block_proposals", Select).value)
            incremental = self.query_one("#use_incremental_scoring", Checkbox).value
            fortran = self.query_one("#use_fortran_batch", Checkbox).value
            transparency = self.query_one("#use_semantic_transparency", Checkbox).value
            t_weight = float(self.query_one("#transparency_weight", Input).value or "0.0")
            snapshot_every = int(self.query_one("#validator_snapshot_every_blocks", Input).value or "0")
            validator_raw = self.query_one("#validator_set", Input).value.strip()
            validator_set = [item.strip() for item in validator_raw.split(",") if item.strip()]
            seed = int(self.query_one("#seed", Input).value or "42")
            return RunConfig(
                source_language="french",
                target_language="latin",
                start_corpus=Path(corpus),
                until_mode="latin_hit",
                candidate_count=n_cand,
                block_proposals=n_block,
                use_incremental_scoring=incremental,
                use_fortran_batch=fortran,
                use_semantic_transparency=transparency,
                transparency_weight=t_weight,
                validator_set=validator_set,
                validator_snapshot_every_blocks=snapshot_every,
                seed=seed,
            )

        def lock_fields(self) -> None:
            """Make all input widgets read-only once the run has started."""
            for widget in self.query(Input):
                widget.disabled = True
            for widget in self.query(Select):
                widget.disabled = True
            for widget in self.query(Checkbox):
                widget.disabled = True


    class RunStatusPanel(Static):
        """Live score display."""

        DEFAULT_CSS = """
        RunStatusPanel {
            border: solid $success;
            padding: 1 2;
            height: 7;
        }
        """

        struct: reactive[float | None] = reactive(None)
        form: reactive[float | None] = reactive(None)
        align: reactive[float | None] = reactive(None)
        proposals: reactive[int] = reactive(0)
        blocks: reactive[int] = reactive(0)
        history_text: reactive[str] = reactive("")
        live_note: reactive[str] = reactive("")

        DEFAULT_CSS = """
        RunStatusPanel {
            border: solid $success;
            padding: 1 2;
            height: auto;
        }
        """

        def render(self) -> str:
            s = f"{self.struct:.4f}" if self.struct is not None else "N/A"
            f_ = f"{self.form:.4f}" if self.form is not None else "N/A"
            a = f"{self.align:.4f}" if self.align is not None else "N/A"
            lines = [
                f"Scores  (proposals={self.proposals:,}  blocks={self.blocks})",
                f"  Structural : {s}  (target >= 0.0)",
                f"  Form       : {f_}  (target >= 1.0)",
                f"  Alignment  : {a}",
            ]
            if self.live_note:
                lines.append(f"  Status     : {self.live_note}")
            if self.history_text:
                lines.append("")
                lines.append(self.history_text)
            return "\n".join(lines)

        def update_status(self, status: RunStatus, history_text: str = "") -> None:
            self.struct = status.last_struct
            self.form = status.last_form
            self.align = status.last_align
            self.proposals = status.cumulative_proposals
            self.blocks = status.blocks_completed
            self.history_text = history_text
            if status.status == "running":
                if status.blocks_completed <= 0 and status.cumulative_proposals <= 0:
                    self.live_note = "first block in progress"
                else:
                    self.live_note = "running"
            elif status.status == "failed" and status.last_error:
                self.live_note = status.last_error
            else:
                self.live_note = status.status

        def update_live_progress(
            self,
            status: RunStatus,
            *,
            current_block_proposals: int,
            current_block_name: str | None = None,
            live_struct: float | None = None,
            live_form: float | None = None,
            live_align: float | None = None,
        ) -> None:
            self.proposals = status.cumulative_proposals + max(current_block_proposals, 0)
            self.blocks = status.blocks_completed
            if live_struct is not None:
                self.struct = live_struct
            if live_form is not None:
                self.form = live_form
            if live_align is not None:
                self.align = live_align
            block_label = current_block_name or f"block_{status.blocks_completed + 1:04d}"
            self.live_note = f"{block_label} in progress"


    class ThroughputPanel(Static):
        """Wall-clock throughput summary."""

        DEFAULT_CSS = """
        ThroughputPanel {
            border: solid $warning;
            padding: 1 2;
            height: auto;
        }
        """

        content: reactive[str] = reactive("  warming up...")

        def render(self) -> str:
            return "Throughput\n" + self.content

        def update_metrics(
            self,
            *,
            elapsed_seconds: float,
            proposals_per_hour: float | None,
            total_proposals: int,
            blocks_completed: int,
            current_block_name: str | None,
        ) -> None:
            pace_text = "warming up first block..." if proposals_per_hour is None else f"{proposals_per_hour:,.0f} p/h"
            lines = [
                f"Wall clock : {_format_hms(elapsed_seconds)}",
                f"Pace       : {pace_text}",
                f"Progress   : {total_proposals:,} proposals across {blocks_completed} completed blocks",
            ]
            if current_block_name:
                lines.append(f"State      : {current_block_name} in progress")
            self.content = "\n".join(lines)


    class LogansRunPanel(Static):
        """Rolling live view of proposal candidates and accepted moves."""

        DEFAULT_CSS = """
        LogansRunPanel {
            border: solid $success;
            padding: 1 2;
            height: auto;
        }
        """

        content: reactive[str] = reactive("  (waiting for first live candidate events...)")

        def render(self) -> str:
            return "Logan's Run\n" + self.content

        def update_from_record(self, record: dict) -> None:
            op = record.get("mutation_operator", "?")
            details = record.get("mutation_details", "")
            stage = record.get("stage_id", "?")
            diag = record.get("diagnostics", {})
            form = diag.get("latin_form_score") or record.get("latin_form_score") or 0.0
            total = record.get("total_score") or 0.0
            struct = record.get("latin_structural_score") or 0.0
            align = diag.get("family_alignment_score") or 0.0

            lines = [
                f"  Stage : {stage}",
                f"  Op    : {op}",
                f"  Scores: form={form:.4f}  struct={struct:.4f}  align={align:.4f}  total={total:.4f}",
            ]
            if details:
                detail_str = str(details)[:200]
                lines.append(f"  Move  : {detail_str}")
            self.content = "\n".join(lines)

        def update_from_events(self, events: list[dict]) -> None:
            lines: list[str] = []
            if not events:
                lines.append("  (waiting for first live candidate events...)")
                self.content = "\n".join(lines)
                return

            for event in events[-10:]:
                proposal = int(event.get("proposal_index", 0))
                candidate = int(event.get("candidate_index", 0))
                outcome = str(event.get("outcome", "?"))
                operator = str(event.get("operator", "?"))
                delta = event.get("score_delta")
                delta_text = "n/a" if delta is None else f"{float(delta):+0.4f}"
                backend = str(event.get("backend", "?"))
                stamp = "--:--:--"
                timestamp = event.get("timestamp_utc")
                if timestamp:
                    stamp = str(timestamp)[11:19]
                details = str(event.get("details", "")).replace("\n", " ")
                details = details[:100]
                lines.append(
                    f"  {stamp} p{proposal:06d} c{candidate:02d} {outcome:<14} {operator:<24} "
                    f"Î”={delta_text:<8} [{backend}] {details}"
                )
            self.content = "\n".join(lines)


    class LiveMovePanel(Static):
        """Live comparison versus frozen v4 baseline."""

        DEFAULT_CSS = """
        LiveMovePanel {
            border: solid $primary;
            padding: 1 2;
            height: auto;
        }
        """

        content: reactive[str] = reactive("  waiting for first comparison sample...")

        def render(self) -> str:
            return "v4 Comparison\n" + self.content

        def update_lines(self, lines: list[str]) -> None:
            self.content = "\n".join(lines) if lines else "  waiting for first comparison sample..."

        def update_from_record(self, record: dict) -> None:
            op = record.get("mutation_operator", "?")
            details = record.get("mutation_details", "")
            stage = record.get("stage_id", "?")
            diag = record.get("diagnostics", {})
            form = diag.get("latin_form_score") or record.get("latin_form_score") or 0.0
            total = record.get("total_score") or 0.0
            struct = record.get("latin_structural_score") or 0.0
            align = diag.get("family_alignment_score") or 0.0

            lines = [
                f"  Stage : {stage}",
                f"  Op    : {op}",
                f"  Scores: form={form:.4f}  struct={struct:.4f}  align={align:.4f}  total={total:.4f}",
            ]
            if details:
                # Truncate long details to fit terminal width
                detail_str = str(details)[:200]
                lines.append(f"  Move  : {detail_str}")
            self.content = "\n".join(lines)


    class ValidatorBankPanel(Static):
        """Latest attested-validator comparison summary."""

        DEFAULT_CSS = """
        ValidatorBankPanel {
            border: solid $accent;
            padding: 1 2;
            height: auto;
        }
        """

        content: reactive[str] = reactive("  Status    : no validator-bank artifacts yet")

        def render(self) -> str:
            return "Validator Bank\n" + self.content

        def update_lines(self, lines: list[str]) -> None:
            self.content = "\n".join(lines) if lines else "  Status    : no validator-bank artifacts yet"

        def update_from_events(self, events: list[dict]) -> None:
            lines: list[str] = []
            if not events:
                lines.append("  (waiting for first live candidate events...)")
                self.content = "\n".join(lines)
                return

            for event in events[-10:]:
                proposal = int(event.get("proposal_index", 0))
                candidate = int(event.get("candidate_index", 0))
                outcome = str(event.get("outcome", "?"))
                operator = str(event.get("operator", "?"))
                delta = event.get("score_delta")
                delta_text = "n/a" if delta is None else f"{float(delta):+0.4f}"
                backend = str(event.get("backend", "?"))
                stamp = "--:--:--"
                timestamp = event.get("timestamp_utc")
                if timestamp:
                    stamp = str(timestamp)[11:19]
                details = str(event.get("details", "")).replace("\n", " ")
                details = details[:100]
                lines.append(
                    f"  {stamp} p{proposal:06d} c{candidate:02d} {outcome:<14} {operator:<24} "
                    f"Δ={delta_text:<8} [{backend}] {details}"
                )
            self.content = "\n".join(lines)


    class LaunchScreen(Screen):
        """Primary launch screen: config + launch button."""

        BINDINGS = [
            Binding("ctrl+c", "app.quit", "Quit"),
        ]

        def compose(self) -> ComposeResult:
            yield Header(show_clock=True)
            yield ConfigPanel(id="config_panel")
            yield Horizontal(
                Button("Launch", id="btn_launch", variant="success"),
                Button("Quit", id="btn_quit", variant="error"),
            )
            yield Footer()

        def on_button_pressed(self, event: Button.Pressed) -> None:
            if event.button.id == "btn_quit":
                self.app.exit()
            elif event.button.id == "btn_launch":
                self._do_launch()

        def _do_launch(self) -> None:
            panel = self.query_one(ConfigPanel)
            try:
                config = panel.build_config()
            except Exception as exc:
                self.notify(f"Config error: {exc}", severity="error")
                return

            errors = config.validate()
            if errors:
                self.notify("\n".join(errors), severity="error")
                return

            for w in config.warnings():
                self.notify(w, severity="warning")

            output_dir = Path(self.query_one("#output_dir", Input).value.strip())
            panel.lock_fields()
            self.app.push_screen(LiveScreen(config=config, output_dir=output_dir))


    class LiveScreen(Screen):
        """Live run monitoring screen."""

        BINDINGS = [
            Binding("s", "stop_run", "Stop at next block"),
            Binding("v", "validate_run", "Validator bank"),
            Binding("ctrl+c", "app.quit", "Quit"),
        ]

        def __init__(self, config: RunConfig, output_dir: Path, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self._config = config
            self._output_dir = output_dir
            self._controller = RunController()
            self._start_time = time.monotonic()
            self._last_proposals = 0
            self._last_blocks = -1
            self._block_completion_time: float | None = None  # monotonic when last block finished
            self._latest_status: RunStatus | None = None
            self._last_status_fingerprint: tuple[Any, ...] | None = None
            self._launch_error: str | None = None
            self._v4_baseline = _load_v4_baseline_reference()
            self._event_accumulator = _LiveEventAccumulator()
            self._validator_compare_running = False
            self._validator_compare_status: str | None = None

        def _current_block_name(self) -> str:
            completed = self._latest_status.blocks_completed if self._latest_status is not None else 0
            return f"block_{completed + 1:04d}"

        def _update_throughput_panel(self, current_block_proposals: int = 0) -> None:
            elapsed = time.monotonic() - self._start_time
            total_proposals = 0
            blocks_completed = 0
            current_block_name: str | None = None
            if self._latest_status is not None:
                total_proposals = self._latest_status.cumulative_proposals + max(current_block_proposals, 0)
                blocks_completed = self._latest_status.blocks_completed
                current_block_name = self._current_block_name()
            current_ph = self._current_average_proposals_per_hour(current_block_proposals)
            self.query_one(ThroughputPanel).update_metrics(
                elapsed_seconds=elapsed,
                proposals_per_hour=current_ph,
                total_proposals=total_proposals,
                blocks_completed=blocks_completed,
                current_block_name=current_block_name,
            )

        def compose(self) -> ComposeResult:
            yield Header(show_clock=True)
            yield RunStatusPanel(id="score_panel")
            yield ThroughputPanel(id="throughput_panel")
            yield LogansRunPanel(id="logans_run_panel")
            yield LiveMovePanel(id="move_panel")
            yield ValidatorBankPanel(id="validator_panel")
            yield Log(id="log_panel", highlight=True)
            yield Footer()

        def on_mount(self) -> None:
            self._launch_thread()
            self._poll_status()
            self._poll_latest_move()
            self._refresh_validator_bank_panel()
            self.set_interval(1.0, self._tick_elapsed)
            self.set_interval(5.0, self._poll_status)
            self.set_interval(3.0, self._poll_latest_move)
            self.set_interval(10.0, self._refresh_validator_bank_panel)

        def _launch_thread(self) -> None:
            thread = threading.Thread(target=self._run_driver, daemon=True)
            thread.start()

        def _run_driver(self) -> None:
            try:
                self._controller.launch_run(self._config, self._output_dir)
            except Exception as exc:
                self._launch_error = f"{type(exc).__name__}: {exc}"
                log.error("Run driver error: %s", exc, exc_info=True)

        def _poll_status(self) -> None:
            """Poll the manifest every 5 s from Textual's main thread."""
            now = time.monotonic()
            status = self._controller.status(self._output_dir)
            self._latest_status = status
            fingerprint = (
                status.status,
                status.cumulative_proposals,
                status.blocks_completed,
                status.last_struct,
                status.last_form,
                status.last_align,
                status.last_updated_utc,
                status.last_error,
            )

            if status.blocks_completed != self._last_blocks:
                # New block completed — compute p/h from inter-block interval
                if self._block_completion_time is not None and self._last_proposals > 0:
                    dt = now - self._block_completion_time
                    dp = status.cumulative_proposals - self._last_proposals
                    if dt > 0 and dp > 0:
                        pass
                self._block_completion_time = now
                self._last_proposals = status.cumulative_proposals
                self._last_blocks = status.blocks_completed
            if fingerprint != self._last_status_fingerprint:
                self._apply_status(status)
                self._last_status_fingerprint = fingerprint
            self._update_throughput_panel()

        def _current_average_proposals_per_hour(self, current_block_proposals: int = 0) -> float | None:
            elapsed = time.monotonic() - self._start_time
            if elapsed <= 0 or self._latest_status is None:
                return None
            total = self._latest_status.cumulative_proposals + max(current_block_proposals, 0)
            if total <= 0:
                return None
            return total / elapsed * 3600.0

        def _build_logans_run_comparison(self, current_block_proposals: int = 0) -> list[str]:
            elapsed = time.monotonic() - self._start_time
            lines = [f"Wall clock : {_format_hms(elapsed)}"]

            current_ph = self._current_average_proposals_per_hour(current_block_proposals)
            current_total = None
            if self._latest_status is not None:
                current_total = self._latest_status.cumulative_proposals + max(current_block_proposals, 0)

            if current_ph is not None and self._v4_baseline is not None:
                pace_lift = _relative_lift(current_ph, self._v4_baseline.proposals_per_hour)
                pace_text = "n/a" if pace_lift is None else f"{pace_lift:+.1f}%"
                lines.append(
                    f"Pace      : {current_ph:,.0f} p/h vs v4 {self._v4_baseline.proposals_per_hour:,.0f} ({pace_text})"
                )
                if current_total is not None:
                    v4_expected = elapsed * self._v4_baseline.proposals_per_hour / 3600.0
                    lines.append(
                        f"Progress  : actual {current_total:,.0f} vs v4-time {v4_expected:,.0f} ({current_total - v4_expected:+,.0f})"
                    )
            elif current_ph is not None:
                lines.append(f"Pace      : {current_ph:,.0f} p/h")
            else:
                lines.append("Pace      : warming up first block...")

            aggregate = self._event_accumulator.aggregate_mean_score_delta()
            if aggregate is not None and self._v4_baseline is not None:
                lift = _relative_lift(aggregate, self._v4_baseline.aggregate_mean_score_delta)
                lift_text = "n/a" if lift is None else f"{lift:+.1f}%"
                lines.append(
                    f"Aggregate : dscore {aggregate:+.4f} vs v4 {self._v4_baseline.aggregate_mean_score_delta:+.4f} ({lift_text})"
                )
            elif aggregate is not None:
                lines.append(f"Aggregate : dscore {aggregate:+.4f}")

            op_rows = self._event_accumulator.top_operator_means(limit=4)
            if op_rows:
                lines.append("Ops vs v4")
                for operator, mean_delta, count in op_rows:
                    label = operator[:22]
                    if self._v4_baseline is not None and operator in self._v4_baseline.operator_mean_score_delta:
                        baseline = self._v4_baseline.operator_mean_score_delta[operator]
                        lift = _relative_lift(mean_delta, baseline)
                        lift_text = "n/a" if lift is None else f"{lift:+.1f}%"
                        lines.append(
                            f"  {label:<22} {mean_delta:+.4f} vs {baseline:+.4f} ({lift_text}) [{count}]"
                        )
                    else:
                        lines.append(f"  {label:<22} {mean_delta:+.4f} [{count}]")

            return lines

        def _apply_status(self, status: RunStatus) -> None:
            manifest_path = self._output_dir / "manifest.json"
            history_text = ""
            if manifest_path.exists():
                try:
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    rows = []
                    for b in manifest.get("blocks", []):
                        name = b["block"].replace("block_", "")
                        form = b.get("final_latin_form_score") or 0.0
                        struct = b.get("final_latin_structural_score") or 0.0
                        align = b.get("final_family_alignment_score") or 0.0
                        rows.append(f"  {name:<8} form={form:.4f}  struct={struct:.4f}  align={align:.4f}")
                    if rows:
                        history_text = "  Block    Form      Struct     Align\n" + "\n".join(rows[-10:])
                except Exception:
                    pass

            self.query_one(RunStatusPanel).update_status(status, history_text)
            self.query_one("#log_panel", Log).write_line(status.summary_line())
            if status.last_error:
                self.query_one("#log_panel", Log).write_line(f"[error] {status.last_error}")

        def _poll_latest_move(self) -> None:
            """Tail the live event stream for the current block."""
            manifest_path = self._output_dir / "manifest.json"
            events = self._event_accumulator.ingest_run_dir(self._output_dir)
            comparison_lines = self._build_logans_run_comparison()
            logans_lines = list(comparison_lines)
            error_text = self._launch_error or (self._latest_status.last_error if self._latest_status is not None else None)
            if error_text:
                logans_lines.extend(["", f"Error     : {error_text}"])
            elif not events and (
                self._latest_status is None
                or (
                    self._latest_status.cumulative_proposals <= 0
                    and self._latest_status.blocks_completed <= 0
                )
            ):
                logans_lines.extend(["", "State     : first block in progress..."])
            self.query_one(LiveMovePanel).update_lines(logans_lines)
            if not manifest_path.exists():
                self._update_throughput_panel()
                self.query_one(LogansRunPanel).update_from_events(events)
                return
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                n_blocks = len(manifest.get("blocks", []))
                # Current block is one ahead of completed blocks
                current_block = f"block_{n_blocks + 1:04d}"
                block_dir = self._output_dir / "blocks" / current_block
                events_path = block_dir / "live_events.jsonl"
                records_dir = block_dir / "records"
                latest_record = None
                if records_dir.exists():
                    files = sorted(records_dir.iterdir())
                    if files:
                        latest_record = json.loads(files[-1].read_text(encoding="utf-8"))
                if events_path.exists():
                    lines = [line for line in events_path.read_text(encoding="utf-8").splitlines() if line.strip()]
                    current_events = [json.loads(line) for line in lines[-10:]]
                    current_block_proposals = 0
                    if current_events:
                        current_block_proposals = int(current_events[-1].get("proposal_index", 0))
                    self._update_throughput_panel(current_block_proposals)
                    if self._latest_status is not None:
                        live_struct = None
                        live_form = None
                        live_align = None
                        if latest_record is not None:
                            diagnostics = latest_record.get("diagnostics", {})
                            live_struct = latest_record.get("latin_structural_score")
                            live_form = diagnostics.get("latin_form_score", latest_record.get("latin_form_score"))
                            live_align = diagnostics.get("family_alignment_score")
                        self.query_one(RunStatusPanel).update_live_progress(
                            self._latest_status,
                            current_block_proposals=current_block_proposals,
                            current_block_name=current_block,
                            live_struct=live_struct,
                            live_form=live_form,
                            live_align=live_align,
                        )
                    comparison_lines = self._build_logans_run_comparison(current_block_proposals)
                    logans_lines = list(comparison_lines)
                    if error_text:
                        logans_lines.extend(["", f"Error     : {error_text}"])
                    elif not current_events and not events and (
                        self._latest_status is None
                        or (
                            self._latest_status.cumulative_proposals <= 0
                            and self._latest_status.blocks_completed <= 0
                        )
                    ):
                        logans_lines.extend(["", "State     : first block in progress..."])
                    self.query_one(LiveMovePanel).update_lines(logans_lines)
                    self.query_one(LogansRunPanel).update_from_events(current_events)
                    return

                if latest_record is not None:
                    if self._latest_status is not None:
                        diagnostics = latest_record.get("diagnostics", {})
                        self.query_one(RunStatusPanel).update_live_progress(
                            self._latest_status,
                            current_block_proposals=0,
                            current_block_name=current_block,
                            live_struct=latest_record.get("latin_structural_score"),
                            live_form=diagnostics.get("latin_form_score", latest_record.get("latin_form_score")),
                            live_align=diagnostics.get("family_alignment_score"),
                        )
                    self._update_throughput_panel()
                    self.query_one(LogansRunPanel).update_from_record(latest_record)
                    return

                self._update_throughput_panel()
                self.query_one(LogansRunPanel).update_from_events(events)
            except Exception:
                log.debug("Could not update live move panel.", exc_info=True)

        def _refresh_validator_bank_panel(self) -> None:
            summary_path = _find_latest_validator_summary(self._output_dir)
            summary = _load_validator_summary(summary_path)
            self.query_one(ValidatorBankPanel).update_lines(
                _validator_bank_lines(
                    summary,
                    summary_path=summary_path,
                    status_text=self._validator_compare_status,
                )
            )

        def _run_validator_compare(self) -> None:
            try:
                result = self._controller.validate_run(self._output_dir)
                self.call_from_thread(self._on_validator_compare_complete, result)
            except Exception as exc:
                self.call_from_thread(self._on_validator_compare_failed, exc)

        def _on_validator_compare_complete(self, result: dict[str, Any]) -> None:
            self._validator_compare_running = False
            self._validator_compare_status = f"updated {datetime.now().strftime('%H:%M:%S')}"
            self._refresh_validator_bank_panel()
            self.query_one("#log_panel", Log).write_line(
                f"[validator] Chronology -> {result.get('summary_path')}"
            )
            self.notify("Validator-bank comparison finished.", severity="information")

        def _on_validator_compare_failed(self, exc: Exception) -> None:
            self._validator_compare_running = False
            self._validator_compare_status = f"error: {type(exc).__name__}"
            self._refresh_validator_bank_panel()
            self.query_one("#log_panel", Log).write_line(
                f"[validator][error] {type(exc).__name__}: {exc}"
            )
            self.notify(f"Validator-bank comparison failed: {exc}", severity="error")

        def _tick_elapsed(self) -> None:
            self._update_throughput_panel()

        def action_stop_run(self) -> None:
            self._controller.stop_run(self._output_dir)
            self.notify("Stop requested. Run will halt at next block boundary.", severity="warning")

        def action_validate_run(self) -> None:
            if self._validator_compare_running:
                self.notify("Validator-bank comparison already running.", severity="warning")
                return
            self._validator_compare_running = True
            self._validator_compare_status = "running full-bank compare..."
            self._refresh_validator_bank_panel()
            self.query_one("#log_panel", Log).write_line("[validator] Starting full-bank compare...")
            thread = threading.Thread(target=self._run_validator_compare, daemon=True)
            thread.start()


    class RetrodactTUI(App):
        """Root TUI application."""

        CSS = """
        Screen {
            background: $background;
        }
        Horizontal {
            height: auto;
            margin: 1 0;
        }
        Button {
            margin: 0 1;
        }
        """

        TITLE = "Project RBT — V5 Retrodiction"
        SUB_TITLE = "French → Latin Bridge Search"

        def on_mount(self) -> None:
            self.push_screen(LaunchScreen())


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    if not _TEXTUAL_AVAILABLE:
        _no_textual()
        return
    app = RetrodactTUI()
    app.run()


if __name__ == "__main__":
    main()
