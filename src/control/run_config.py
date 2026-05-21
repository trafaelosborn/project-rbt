"""
V5 Shared Run Configuration
============================
Purpose:
    One config model that powers CLI, TUI, and tests alike.

    Settings are validated at construction time. After a run is launched the
    config is locked via `lock()` — any further mutation raises `ConfigLockedError`.

    Bounded presets are enforced for candidate_count and block_proposals so that
    the search space is reproducible and benchmarkable. Values outside those
    presets are rejected at construction, not silently rounded.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Bounded presets
# ---------------------------------------------------------------------------

CANDIDATE_PRESETS: tuple[int, ...] = (8, 16, 32, 64, 100)
BLOCK_PROPOSALS_PRESETS: tuple[int, ...] = (1000, 5000)
UNTIL_MODES: tuple[str, ...] = ("latin_hit", "plateau", "budget")
TARGET_LANGUAGES: tuple[str, ...] = ("latin",)
SOURCE_LANGUAGES: tuple[str, ...] = ("french", "spanish", "portuguese", "italian", "romanian")
LIVE_EVENT_MODES: tuple[str, ...] = ("all", "selected", "accepted_only", "off")
RECOMMENDED_V5_CANDIDATE_COUNT: int = 16
HIGH_DEPTH_V5_CANDIDATE_COUNT: int = 100
RECOMMENDED_V5_USE_FORTRAN_BATCH: bool = True


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class ConfigLockedError(RuntimeError):
    """Raised when a locked RunConfig is mutated."""


class ConfigValidationError(ValueError):
    """Raised when a RunConfig fails validation."""


# ---------------------------------------------------------------------------
# Config model
# ---------------------------------------------------------------------------

@dataclass
class RunConfig:
    """
    Launch-time configuration for a v5 retrodiction run.

    This is the single source of truth shared by CLI, TUI, and the
    RunController. Once `lock()` is called (at run launch), all fields
    become read-only.

    Experimental conditions
    -----------------------
    `use_semantic_transparency` and `use_fortran_batch` are *experimental
    conditions*, not merely UX toggles. Changing them mid-run would invalidate
    the run as a controlled experiment, which is why the lock exists.

    Fortran batch status
    --------------------
    `use_fortran_batch=True` enables the Fortran-backed batch candidate scoring
    path inside the proposal loop. The current implementation uses the compiled
    cosine layer for batched form scoring and a specialized batched
    structural/coherence selection path around it, while accepted-stage
    fingerprint artifact generation remains in Python. This keeps methodology
    unchanged while reducing per-proposal scoring overhead.
    """

    # --- identity ---
    source_language: str = "french"
    target_language: str = "latin"

    # --- corpus ---
    start_corpus: Path = field(default_factory=lambda: Path(""))

    # --- stopping ---
    until_mode: str = "latin_hit"
    total_target_proposals: int = 0       # 0 = unlimited (used when mode != "budget")
    struct_target: float = 0.0
    form_target: float = 1.0

    # --- search breadth ---
    candidate_count: int = RECOMMENDED_V5_CANDIDATE_COUNT
    block_proposals: int = 1000           # must be in BLOCK_PROPOSALS_PRESETS
    num_sequences: int = 800
    seed: int = 42

    # --- experimental conditions (locked at launch) ---
    use_incremental_scoring: bool = True
    use_fortran_cosine: bool = True          # Fortran/BLAS dense cosine in incremental scoring path
    use_fortran_batch: bool = RECOMMENDED_V5_USE_FORTRAN_BATCH
    use_semantic_transparency: bool = False
    transparency_weight: float = 0.0

    # --- validation ---
    validator_set: list[str] = field(default_factory=list)
    validator_snapshot_every_blocks: int = 0

    # --- live stream / logging ---
    live_event_mode: str = "all"
    live_event_buffer_size: int = 64

    # --- internal lock (not serialised) ---
    _locked: bool = field(default=False, init=False, repr=False, compare=False)

    # ------------------------------------------------------------------
    # Lock / mutation guard
    # ------------------------------------------------------------------

    def lock(self) -> None:
        """Freeze this config. Called once at run launch; cannot be undone."""
        object.__setattr__(self, "_locked", True)

    @property
    def is_locked(self) -> bool:
        return self._locked

    def __setattr__(self, name: str, value: Any) -> None:
        if name != "_locked" and getattr(self, "_locked", False):
            raise ConfigLockedError(
                f"RunConfig is locked after run launch. Cannot set {name!r}."
            )
        object.__setattr__(self, name, value)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> list[str]:
        """
        Return a list of validation error strings.
        Empty list means the config is valid.
        """
        errors: list[str] = []

        if self.source_language not in SOURCE_LANGUAGES:
            errors.append(
                f"source_language {self.source_language!r} not in {SOURCE_LANGUAGES}"
            )

        if self.target_language not in TARGET_LANGUAGES:
            errors.append(
                f"target_language {self.target_language!r} not in {TARGET_LANGUAGES}"
            )

        corpus_path = Path(self.start_corpus)
        if not corpus_path.exists():
            errors.append(f"start_corpus does not exist: {corpus_path}")

        if self.until_mode not in UNTIL_MODES:
            errors.append(
                f"until_mode {self.until_mode!r} not in {UNTIL_MODES}"
            )

        if self.live_event_mode not in LIVE_EVENT_MODES:
            errors.append(
                f"live_event_mode {self.live_event_mode!r} not in {LIVE_EVENT_MODES}"
            )

        if self.candidate_count not in CANDIDATE_PRESETS:
            errors.append(
                f"candidate_count {self.candidate_count} not in CANDIDATE_PRESETS {CANDIDATE_PRESETS}"
            )

        if self.block_proposals not in BLOCK_PROPOSALS_PRESETS:
            errors.append(
                f"block_proposals {self.block_proposals} not in BLOCK_PROPOSALS_PRESETS {BLOCK_PROPOSALS_PRESETS}"
            )

        if self.num_sequences < 10:
            errors.append(f"num_sequences {self.num_sequences} is implausibly small (< 10)")

        if self.until_mode == "budget" and self.total_target_proposals <= 0:
            errors.append(
                "until_mode='budget' requires total_target_proposals > 0"
            )

        if self.use_semantic_transparency and self.transparency_weight <= 0.0:
            errors.append(
                "use_semantic_transparency=True requires transparency_weight > 0.0 "
                "(calibrate in a probe run first)"
            )

        if self.validator_snapshot_every_blocks < 0:
            errors.append("validator_snapshot_every_blocks must be >= 0")

        if self.live_event_buffer_size < 1:
            errors.append("live_event_buffer_size must be >= 1")

        return errors

    def validate_strict(self) -> None:
        """Raise ConfigValidationError if any validation errors exist."""
        errors = self.validate()
        if errors:
            raise ConfigValidationError(
                "RunConfig validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
            )

    def warnings(self) -> list[str]:
        """
        Return non-fatal warnings. The run can proceed but the operator
        should be aware of these conditions.
        """
        warns: list[str] = []

        if self.use_fortran_batch:
            warns.append(
                "use_fortran_batch=True enables the Fortran-backed batch candidate scoring path. "
                "Treat this as an experimental acceleration condition and benchmark it "
                "against the Python baseline from the same start corpus."
            )

        if self.use_semantic_transparency:
            warns.append(
                "use_semantic_transparency=True is an experimental condition. "
                "Results should be reported alongside a transparency=False ablation."
            )

        if self.block_proposals == 5000 and self.until_mode != "budget":
            warns.append(
                "block_proposals=5000 with unbounded run: intermediate status will "
                "only be visible at block boundaries."
            )

        if self.live_event_mode == "all":
            warns.append(
                "live_event_mode='all' writes every candidate event. This is best for "
                "full Logan's Run visibility, but can cost throughput on long runs."
            )

        if self.validator_snapshot_every_blocks > 0:
            validator_label = ", ".join(self.validator_set) if self.validator_set else "all active attested validators"
            warns.append(
                "validator_snapshot_every_blocks>0 enables block-level validator-bank "
                f"snapshots against {validator_label}. This adds science outputs at "
                "block boundaries without changing hot-loop methodology."
            )

        return warns

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_language": self.source_language,
            "target_language": self.target_language,
            "start_corpus": str(self.start_corpus),
            "until_mode": self.until_mode,
            "total_target_proposals": self.total_target_proposals,
            "struct_target": self.struct_target,
            "form_target": self.form_target,
            "candidate_count": self.candidate_count,
            "block_proposals": self.block_proposals,
            "num_sequences": self.num_sequences,
            "seed": self.seed,
            "use_incremental_scoring": self.use_incremental_scoring,
            "use_fortran_cosine": self.use_fortran_cosine,
            "use_fortran_batch": self.use_fortran_batch,
            "use_semantic_transparency": self.use_semantic_transparency,
            "transparency_weight": self.transparency_weight,
            "validator_set": list(self.validator_set),
            "validator_snapshot_every_blocks": self.validator_snapshot_every_blocks,
            "live_event_mode": self.live_event_mode,
            "live_event_buffer_size": self.live_event_buffer_size,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunConfig":
        known = {f.name for f in fields(cls) if f.name != "_locked"}
        filtered = {k: v for k, v in data.items() if k in known}
        if "start_corpus" in filtered:
            filtered["start_corpus"] = Path(filtered["start_corpus"])
        return cls(**filtered)

    @classmethod
    def from_json(cls, path: Path) -> "RunConfig":
        with path.open(encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Benchmark config (separate; no lock needed)
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkConfig:
    """Config for a candidate-count or block-size benchmark sweep."""

    source_language: str = "french"
    target_language: str = "latin"
    start_corpus: Path = field(default_factory=lambda: Path(""))
    candidate_counts: list[int] = field(default_factory=lambda: list(CANDIDATE_PRESETS))
    proposals_per_trial: int = 200
    num_sequences: int = 800
    seed: int = 42
    output_dir: Path = field(default_factory=lambda: Path("data/benchmarks/v5"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_language": self.source_language,
            "target_language": self.target_language,
            "start_corpus": str(self.start_corpus),
            "candidate_counts": self.candidate_counts,
            "proposals_per_trial": self.proposals_per_trial,
            "num_sequences": self.num_sequences,
            "seed": self.seed,
            "output_dir": str(self.output_dir),
        }
