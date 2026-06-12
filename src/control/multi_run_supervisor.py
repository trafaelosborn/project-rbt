"""
Multi-Run Supervisor
====================
Purpose:
    Launch and monitor a matrix of independent Romance -> Latin long-run jobs
    without letting concurrent processes make the workstation unusable.

The supervisor is intentionally thin. Each child is still a normal
`src.retrodiction.long_run_v5` process with its own manifest, block summaries,
and resumable output directory. This module only schedules those children,
sets conservative process/thread limits, and provides batch status/pause
operations over the manifests.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

from src.control.run_config import (
    BLOCK_PROPOSALS_PRESETS,
    CANDIDATE_PRESETS,
    LIVE_EVENT_MODES,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "v2"
SUPERVISOR_MANIFEST = "supervisor_manifest.json"
PAUSE_SENTINEL = "pause_requested"
STOP_SENTINEL = "stop_requested"

DEFAULT_SOURCE_LANGUAGES = ("french", "spanish", "italian", "portuguese", "romanian")
SUCCESS_STATUSES = {"complete", "joint_hit", "plateau_hit"}
TERMINAL_STATUSES = SUCCESS_STATUSES | {"stopped_manual", "failed"}


@dataclass(frozen=True)
class SupervisorProfile:
    """Resource policy for a batch supervisor run."""

    name: str
    max_parallel: int
    priority: str
    thread_count: int
    live_event_mode: str


PROFILES: dict[str, SupervisorProfile] = {
    "polite": SupervisorProfile(
        name="polite",
        max_parallel=3,
        priority="below_normal",
        thread_count=1,
        live_event_mode="accepted_only",
    ),
    "overnight": SupervisorProfile(
        name="overnight",
        max_parallel=6,
        priority="below_normal",
        thread_count=1,
        live_event_mode="accepted_only",
    ),
    "full": SupervisorProfile(
        name="full",
        max_parallel=6,
        priority="normal",
        thread_count=1,
        live_event_mode="off",
    ),
}


@dataclass(frozen=True)
class SupervisorJob:
    """One child long-run process in a batch."""

    language: str
    seed: int
    start_corpus: Path
    output_dir: Path
    total_target_proposals: int
    block_proposals: int = 1000
    candidate_count: int = 16
    num_sequences: int = 800
    use_fortran_cosine: bool = True
    use_fortran_batch: bool = True
    live_event_mode: str = "accepted_only"
    live_event_buffer_size: int = 64
    min_improvement: float = 0.0001
    struct_target: float = 0.0
    form_target: float = 1.0
    family_target: float = 1.0
    validator_set: tuple[str, ...] = field(default_factory=tuple)
    validator_snapshot_every_blocks: int = 0
    features_freeze_commit: str = ""
    features_freeze_sha256: str = ""
    corpus_budget: int = 20_000

    @property
    def job_id(self) -> str:
        return f"{self.language}_seed_{self.seed}"

    @property
    def manifest_path(self) -> Path:
        return self.output_dir / "manifest.json"

    @property
    def stop_sentinel(self) -> Path:
        return self.output_dir / STOP_SENTINEL

    def command(self) -> list[str]:
        cmd = [
            sys.executable,
            "-m",
            "src.retrodiction.long_run_v5",
            "--language",
            self.language,
            "--start-corpus",
            str(self.start_corpus),
            "--output-dir",
            str(self.output_dir),
            "--target-proposals",
            str(self.total_target_proposals),
            "--block-proposals",
            str(self.block_proposals),
            "--num-sequences",
            str(self.num_sequences),
            "--n-candidates",
            str(self.candidate_count),
            "--max-accepted-stages",
            "512",
            "--seed",
            str(self.seed),
            "--min-improvement",
            str(self.min_improvement),
            "--struct-target",
            str(self.struct_target),
            "--form-target",
            str(self.form_target),
            "--family-target",
            str(self.family_target),
            "--live-event-mode",
            self.live_event_mode,
            "--live-event-buffer-size",
            str(self.live_event_buffer_size),
            "--plateau-window-blocks",
            "10",
            "--plateau-struct-epsilon",
            "0.001",
            "--plateau-form-epsilon",
            "0.001",
            "--plateau-family-epsilon",
            "0.001",
        ]

        cmd.append("--use-fortran-cosine" if self.use_fortran_cosine else "--no-use-fortran-cosine")
        cmd.append("--use-fortran-batch" if self.use_fortran_batch else "--no-use-fortran-batch")

        if self.validator_set:
            cmd.append("--validator-set")
            cmd.extend(self.validator_set)
        if self.validator_snapshot_every_blocks > 0:
            cmd.extend([
                "--validator-snapshot-every-blocks",
                str(self.validator_snapshot_every_blocks),
            ])
        if self.features_freeze_commit:
            cmd.extend(["--features-freeze-commit", self.features_freeze_commit])
        if self.features_freeze_sha256:
            cmd.extend(["--features-freeze-sha256", self.features_freeze_sha256])
        cmd.extend(["--corpus-budget", str(self.corpus_budget)])

        return cmd

    def to_dict(self) -> dict[str, Any]:
        return {
            "language": self.language,
            "seed": self.seed,
            "start_corpus": str(self.start_corpus),
            "output_dir": str(self.output_dir),
            "total_target_proposals": self.total_target_proposals,
            "block_proposals": self.block_proposals,
            "candidate_count": self.candidate_count,
            "num_sequences": self.num_sequences,
            "use_fortran_cosine": self.use_fortran_cosine,
            "use_fortran_batch": self.use_fortran_batch,
            "live_event_mode": self.live_event_mode,
            "live_event_buffer_size": self.live_event_buffer_size,
            "min_improvement": self.min_improvement,
            "struct_target": self.struct_target,
            "form_target": self.form_target,
            "family_target": self.family_target,
            "validator_set": list(self.validator_set),
            "validator_snapshot_every_blocks": self.validator_snapshot_every_blocks,
            "features_freeze_commit": self.features_freeze_commit,
            "features_freeze_sha256": self.features_freeze_sha256,
            "corpus_budget": self.corpus_budget,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SupervisorJob":
        return cls(
            language=str(data["language"]),
            seed=int(data["seed"]),
            start_corpus=Path(data["start_corpus"]),
            output_dir=Path(data["output_dir"]),
            total_target_proposals=int(data["total_target_proposals"]),
            block_proposals=int(data.get("block_proposals", 1000)),
            candidate_count=int(data.get("candidate_count", 16)),
            num_sequences=int(data.get("num_sequences", 800)),
            use_fortran_cosine=bool(data.get("use_fortran_cosine", True)),
            use_fortran_batch=bool(data.get("use_fortran_batch", True)),
            live_event_mode=str(data.get("live_event_mode", "accepted_only")),
            live_event_buffer_size=int(data.get("live_event_buffer_size", 64)),
            min_improvement=float(data.get("min_improvement", 0.0001)),
            struct_target=float(data.get("struct_target", 0.0)),
            form_target=float(data.get("form_target", 1.0)),
            family_target=float(data.get("family_target", 1.0)),
            validator_set=tuple(data.get("validator_set", [])),
            validator_snapshot_every_blocks=int(data.get("validator_snapshot_every_blocks", 0)),
            features_freeze_commit=str(data.get("features_freeze_commit", "")),
            features_freeze_sha256=str(data.get("features_freeze_sha256", "")),
            corpus_budget=int(data.get("corpus_budget", 20_000)),
        )


@dataclass(frozen=True)
class JobStatus:
    """Manifest-derived status for one supervisor job."""

    job_id: str
    language: str
    seed: int
    output_dir: Path
    status: str
    cumulative_proposals: int
    blocks_completed: int
    last_struct: float | None
    last_form: float | None
    last_align: float | None
    current_corpus: str | None
    last_error: str | None
    return_code: int | None = None

    @property
    def is_success(self) -> bool:
        return self.status in SUCCESS_STATUSES

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "language": self.language,
            "seed": self.seed,
            "output_dir": str(self.output_dir),
            "status": self.status,
            "cumulative_proposals": self.cumulative_proposals,
            "blocks_completed": self.blocks_completed,
            "last_struct": self.last_struct,
            "last_form": self.last_form,
            "last_align": self.last_align,
            "current_corpus": self.current_corpus,
            "last_error": self.last_error,
            "return_code": self.return_code,
        }


@dataclass
class RunningJob:
    """A started child process plus log handles that must be closed."""

    job: SupervisorJob
    process: subprocess.Popen
    stdout: TextIO
    stderr: TextIO

    def close_logs(self) -> None:
        self.stdout.close()
        self.stderr.close()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_start_corpus(language: str) -> Path:
    """Return the repo-local default token corpus for a source language."""
    processed = PROJECT_ROOT / "data" / "processed" / "romance" / f"{language}_tokens.json"
    if processed.exists():
        return processed
    sequestered = PROJECT_ROOT / "data" / "sequestered" / language / f"{language}_tokens.json"
    return sequestered


def build_jobs(
    *,
    languages: list[str],
    seeds: list[int],
    output_root: Path,
    total_target_proposals: int,
    block_proposals: int = 1000,
    candidate_count: int = 16,
    num_sequences: int = 800,
    live_event_mode: str = "accepted_only",
    use_fortran_batch: bool = True,
    features_freeze_commit: str = "",
    features_freeze_sha256: str = "",
    corpus_budget: int = 20_000,
) -> list[SupervisorJob]:
    """Expand a language x seed matrix into per-run jobs."""
    jobs: list[SupervisorJob] = []
    for language in languages:
        start_corpus = default_start_corpus(language)
        for seed in seeds:
            output_dir = output_root / language / f"seed_{seed}"
            jobs.append(
                SupervisorJob(
                    language=language,
                    seed=seed,
                    start_corpus=start_corpus,
                    output_dir=output_dir,
                    total_target_proposals=total_target_proposals,
                    block_proposals=block_proposals,
                    candidate_count=candidate_count,
                    num_sequences=num_sequences,
                    use_fortran_batch=use_fortran_batch,
                    live_event_mode=live_event_mode,
                    features_freeze_commit=features_freeze_commit,
                    features_freeze_sha256=features_freeze_sha256,
                    corpus_budget=corpus_budget,
                )
            )
    return jobs


def validate_jobs(jobs: list[SupervisorJob]) -> list[str]:
    errors: list[str] = []
    for job in jobs:
        if not job.start_corpus.exists():
            errors.append(f"{job.job_id}: start corpus missing: {job.start_corpus}")
        if job.block_proposals not in BLOCK_PROPOSALS_PRESETS:
            errors.append(f"{job.job_id}: invalid block_proposals={job.block_proposals}")
        if job.candidate_count not in CANDIDATE_PRESETS:
            errors.append(f"{job.job_id}: invalid candidate_count={job.candidate_count}")
        if job.live_event_mode not in LIVE_EVENT_MODES:
            errors.append(f"{job.job_id}: invalid live_event_mode={job.live_event_mode!r}")
        if job.features_freeze_sha256 and len(job.features_freeze_sha256) != 64:
            errors.append(f"{job.job_id}: features_freeze_sha256 must be 64 hex characters")
        if job.corpus_budget <= 0:
            errors.append(f"{job.job_id}: corpus_budget must be positive")
    return errors


def supervisor_manifest_path(output_root: Path) -> Path:
    return output_root / SUPERVISOR_MANIFEST


def write_supervisor_manifest(
    output_root: Path,
    *,
    jobs: list[SupervisorJob],
    profile: SupervisorProfile,
    max_parallel: int,
) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_utc": utc_now_iso(),
        "updated_utc": utc_now_iso(),
        "profile": profile.name,
        "max_parallel": max_parallel,
        "priority": profile.priority,
        "thread_count": profile.thread_count,
        "job_count": len(jobs),
        "jobs": [job.to_dict() for job in jobs],
    }
    path = supervisor_manifest_path(output_root)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    return path


def load_supervisor_manifest(output_root: Path) -> tuple[dict[str, Any], list[SupervisorJob]]:
    path = supervisor_manifest_path(output_root)
    if not path.exists():
        raise FileNotFoundError(f"No supervisor manifest at {path}")
    with path.open(encoding="utf-8") as fh:
        payload = json.load(fh)
    jobs = [SupervisorJob.from_dict(item) for item in payload.get("jobs", [])]
    return payload, jobs


def _last_block(manifest: dict[str, Any]) -> dict[str, Any]:
    blocks = manifest.get("blocks", [])
    if not blocks:
        return {}
    return blocks[-1]


def read_job_status(job: SupervisorJob, return_code: int | None = None) -> JobStatus:
    if not job.manifest_path.exists():
        return JobStatus(
            job_id=job.job_id,
            language=job.language,
            seed=job.seed,
            output_dir=job.output_dir,
            status="not_found",
            cumulative_proposals=0,
            blocks_completed=0,
            last_struct=None,
            last_form=None,
            last_align=None,
            current_corpus=None,
            last_error=None,
            return_code=return_code,
        )

    try:
        with job.manifest_path.open(encoding="utf-8") as fh:
            manifest = json.load(fh)
    except json.JSONDecodeError as exc:
        return JobStatus(
            job_id=job.job_id,
            language=job.language,
            seed=job.seed,
            output_dir=job.output_dir,
            status="invalid_manifest",
            cumulative_proposals=0,
            blocks_completed=0,
            last_struct=None,
            last_form=None,
            last_align=None,
            current_corpus=None,
            last_error=str(exc),
            return_code=return_code,
        )

    blocks = manifest.get("blocks", [])
    last = _last_block(manifest)
    return JobStatus(
        job_id=job.job_id,
        language=job.language,
        seed=job.seed,
        output_dir=job.output_dir,
        status=str(manifest.get("status", "unknown")),
        cumulative_proposals=int(manifest.get("cumulative_proposals", 0)),
        blocks_completed=len(blocks),
        last_struct=last.get("final_latin_structural_score"),
        last_form=last.get("final_latin_form_score"),
        last_align=last.get("final_family_alignment_score"),
        current_corpus=manifest.get("current_corpus"),
        last_error=manifest.get("last_error"),
        return_code=return_code,
    )


def summarize_jobs(jobs: list[SupervisorJob]) -> dict[str, Any]:
    statuses = [read_job_status(job) for job in jobs]
    counts: dict[str, int] = {}
    for status in statuses:
        counts[status.status] = counts.get(status.status, 0) + 1
    return {
        "job_count": len(jobs),
        "counts": counts,
        "total_proposals": sum(status.cumulative_proposals for status in statuses),
        "jobs": [status.to_dict() for status in statuses],
    }


def controlled_environment(thread_count: int) -> dict[str, str]:
    """Environment for one child process with BLAS/OpenMP thread caps."""
    env = os.environ.copy()
    value = str(max(1, int(thread_count)))
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        env[key] = value
    env["PYTHONUNBUFFERED"] = "1"
    return env


def creation_flags_for_priority(priority: str) -> int:
    if os.name != "nt":
        return 0
    flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    priority_map = {
        "idle": getattr(subprocess, "IDLE_PRIORITY_CLASS", 0),
        "below_normal": getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0),
        "normal": getattr(subprocess, "NORMAL_PRIORITY_CLASS", 0),
    }
    return flags | priority_map.get(priority, 0)


def _start_job(job: SupervisorJob, *, profile: SupervisorProfile) -> RunningJob:
    job.output_dir.mkdir(parents=True, exist_ok=True)
    stdout = (job.output_dir / "supervisor_stdout.log").open("a", encoding="utf-8")
    stderr = (job.output_dir / "supervisor_stderr.log").open("a", encoding="utf-8")
    process = subprocess.Popen(
        job.command(),
        cwd=PROJECT_ROOT,
        env=controlled_environment(profile.thread_count),
        stdout=stdout,
        stderr=stderr,
        text=True,
        creationflags=creation_flags_for_priority(profile.priority),
    )
    return RunningJob(job=job, process=process, stdout=stdout, stderr=stderr)


def should_launch(
    status: JobStatus,
    *,
    retry_failed: bool = False,
    resume_running: bool = False,
) -> bool:
    if status.is_success:
        return False
    if status.status == "failed" and not retry_failed:
        return False
    if status.status == "running" and not resume_running:
        return False
    return True


def pause_requested(output_root: Path) -> bool:
    return (output_root / PAUSE_SENTINEL).exists()


def pause_jobs(output_root: Path) -> list[Path]:
    """Request a soft stop for every job in a supervisor manifest."""
    _, jobs = load_supervisor_manifest(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / PAUSE_SENTINEL).touch()
    sentinels: list[Path] = []
    for job in jobs:
        job.output_dir.mkdir(parents=True, exist_ok=True)
        job.stop_sentinel.touch()
        sentinels.append(job.stop_sentinel)
    return sentinels


def clear_pause_requests(output_root: Path) -> int:
    """Remove supervisor and per-job pause sentinels before resuming."""
    removed = 0
    pause_path = output_root / PAUSE_SENTINEL
    if pause_path.exists():
        pause_path.unlink()
        removed += 1
    _, jobs = load_supervisor_manifest(output_root)
    for job in jobs:
        if job.stop_sentinel.exists():
            job.stop_sentinel.unlink()
            removed += 1
    return removed


def run_supervisor(
    *,
    output_root: Path,
    jobs: list[SupervisorJob],
    profile: SupervisorProfile,
    max_parallel: int | None = None,
    poll_seconds: float = 5.0,
    retry_failed: bool = False,
    resume_running: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    errors = validate_jobs(jobs)
    if errors:
        raise ValueError("Invalid supervisor jobs:\n" + "\n".join(f"  - {e}" for e in errors))

    actual_parallel = max(1, int(max_parallel or profile.max_parallel))
    write_supervisor_manifest(
        output_root,
        jobs=jobs,
        profile=profile,
        max_parallel=actual_parallel,
    )
    if dry_run:
        return summarize_jobs(jobs)

    pending = [
        job
        for job in jobs
        if should_launch(
            read_job_status(job),
            retry_failed=retry_failed,
            resume_running=resume_running,
        )
    ]
    running: list[RunningJob] = []

    try:
        while pending or running:
            while pending and len(running) < actual_parallel and not pause_requested(output_root):
                job = pending.pop(0)
                if job.stop_sentinel.exists():
                    job.stop_sentinel.unlink()
                started = _start_job(job, profile=profile)
                running.append(started)
                print(f"started {job.job_id} pid={started.process.pid}", flush=True)

            if not running:
                break

            time.sleep(max(0.1, poll_seconds))
            still_running: list[RunningJob] = []
            for item in running:
                return_code = item.process.poll()
                if return_code is None:
                    still_running.append(item)
                    continue

                status = read_job_status(item.job, return_code=return_code)
                print(
                    f"finished {item.job.job_id} rc={return_code} "
                    f"status={status.status} proposals={status.cumulative_proposals}",
                    flush=True,
                )
                item.close_logs()

            running = still_running

            if pause_requested(output_root) and not running:
                break
    finally:
        for item in running:
            item.close_logs()

    return summarize_jobs(jobs)


def print_status(summary: dict[str, Any]) -> None:
    counts = ", ".join(f"{key}={value}" for key, value in sorted(summary["counts"].items()))
    print(f"jobs={summary['job_count']} proposals={summary['total_proposals']} {counts}")
    for item in summary["jobs"]:
        struct = "N/A" if item["last_struct"] is None else f"{item['last_struct']:.4f}"
        form = "N/A" if item["last_form"] is None else f"{item['last_form']:.4f}"
        align = "N/A" if item["last_align"] is None else f"{item['last_align']:.4f}"
        print(
            f"{item['job_id']:22s} {item['status']:15s} "
            f"blocks={item['blocks_completed']:4d} proposals={item['cumulative_proposals']:8d} "
            f"struct={struct:>8s} form={form:>8s} align={align:>8s}"
        )


def _profile_from_args(args: argparse.Namespace) -> SupervisorProfile:
    base = PROFILES[args.profile]
    max_parallel = args.max_parallel if args.max_parallel is not None else base.max_parallel
    priority = args.priority if args.priority is not None else base.priority
    thread_count = args.thread_count if args.thread_count is not None else base.thread_count
    live_event_mode = args.live_event_mode if args.live_event_mode is not None else base.live_event_mode
    return SupervisorProfile(
        name=base.name,
        max_parallel=max_parallel,
        priority=priority,
        thread_count=thread_count,
        live_event_mode=live_event_mode,
    )


def _jobs_from_args(args: argparse.Namespace, profile: SupervisorProfile) -> list[SupervisorJob]:
    total_target = 0 if args.unbounded else args.target_proposals
    return build_jobs(
        languages=list(args.languages),
        seeds=[int(seed) for seed in args.seeds],
        output_root=args.output_root,
        total_target_proposals=total_target,
        block_proposals=args.block_proposals,
        candidate_count=args.candidate_count,
        num_sequences=args.num_sequences,
        live_event_mode=profile.live_event_mode,
        use_fortran_batch=args.fortran_batch,
        features_freeze_commit=args.features_freeze_commit,
        features_freeze_sha256=args.features_freeze_sha256,
        corpus_budget=args.corpus_budget,
    )


def _add_matrix_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--languages", nargs="+", default=list(DEFAULT_SOURCE_LANGUAGES))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--profile", choices=sorted(PROFILES), default="polite")
    parser.add_argument("--max-parallel", type=int, default=None)
    parser.add_argument("--priority", choices=["idle", "below_normal", "normal"], default=None)
    parser.add_argument("--thread-count", type=int, default=None)
    parser.add_argument("--target-proposals", type=int, default=1000)
    parser.add_argument("--unbounded", action="store_true", help="Run until plateau/joint hit instead of a fixed budget")
    parser.add_argument("--block-proposals", type=int, choices=list(BLOCK_PROPOSALS_PRESETS), default=1000)
    parser.add_argument("--candidate-count", type=int, choices=list(CANDIDATE_PRESETS), default=16)
    parser.add_argument("--num-sequences", type=int, default=800)
    parser.add_argument("--fortran-batch", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--live-event-mode", choices=list(LIVE_EVENT_MODES), default=None)
    parser.add_argument("--features-freeze-commit", default="")
    parser.add_argument("--features-freeze-sha256", default="")
    parser.add_argument("--corpus-budget", type=int, default=20_000)


def cmd_plan(args: argparse.Namespace) -> int:
    profile = _profile_from_args(args)
    jobs = _jobs_from_args(args, profile)
    errors = validate_jobs(jobs)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    path = write_supervisor_manifest(
        args.output_root,
        jobs=jobs,
        profile=profile,
        max_parallel=profile.max_parallel,
    )
    print(f"wrote {path}")
    print_status(summarize_jobs(jobs))
    return 0


def cmd_start(args: argparse.Namespace) -> int:
    profile = _profile_from_args(args)
    jobs = _jobs_from_args(args, profile)
    try:
        summary = run_supervisor(
            output_root=args.output_root,
            jobs=jobs,
            profile=profile,
            max_parallel=profile.max_parallel,
            poll_seconds=args.poll_seconds,
            retry_failed=args.retry_failed,
            resume_running=args.resume_running,
            dry_run=args.dry_run,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print_status(summary)
    return 0


def cmd_resume(args: argparse.Namespace) -> int:
    _, jobs = load_supervisor_manifest(args.output_root)
    removed = clear_pause_requests(args.output_root)
    profile = _profile_from_args(args)
    print(f"cleared {removed} pause sentinels")
    summary = run_supervisor(
        output_root=args.output_root,
        jobs=jobs,
        profile=profile,
        max_parallel=profile.max_parallel,
        poll_seconds=args.poll_seconds,
        retry_failed=args.retry_failed,
        resume_running=args.resume_running,
    )
    print_status(summary)
    return 0


def cmd_pause(args: argparse.Namespace) -> int:
    sentinels = pause_jobs(args.output_root)
    print(f"pause requested for {len(sentinels)} jobs")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    try:
        _, jobs = load_supervisor_manifest(args.output_root)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    summary = summarize_jobs(jobs)
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print_status(summary)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.control.multi_run_supervisor",
        description="Launch and monitor a capped matrix of v5 long-run jobs",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_plan = subparsers.add_parser("plan", help="Write a supervisor manifest without launching jobs")
    _add_matrix_args(p_plan)
    p_plan.set_defaults(func=cmd_plan)

    p_start = subparsers.add_parser("start", help="Create a job matrix and run it")
    _add_matrix_args(p_start)
    p_start.add_argument("--poll-seconds", type=float, default=5.0)
    p_start.add_argument("--retry-failed", action="store_true")
    p_start.add_argument("--resume-running", action="store_true")
    p_start.add_argument("--dry-run", action="store_true")
    p_start.set_defaults(func=cmd_start)

    p_resume = subparsers.add_parser("resume", help="Resume jobs from an existing supervisor manifest")
    p_resume.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p_resume.add_argument("--profile", choices=sorted(PROFILES), default="polite")
    p_resume.add_argument("--max-parallel", type=int, default=None)
    p_resume.add_argument("--priority", choices=["idle", "below_normal", "normal"], default=None)
    p_resume.add_argument("--thread-count", type=int, default=None)
    p_resume.add_argument("--live-event-mode", choices=list(LIVE_EVENT_MODES), default=None)
    p_resume.add_argument("--poll-seconds", type=float, default=5.0)
    p_resume.add_argument("--retry-failed", action="store_true")
    p_resume.add_argument("--resume-running", action="store_true")
    p_resume.set_defaults(func=cmd_resume)

    p_pause = subparsers.add_parser("pause", help="Soft-stop all jobs at their next block boundary")
    p_pause.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p_pause.set_defaults(func=cmd_pause)

    p_status = subparsers.add_parser("status", help="Show all job statuses from manifests")
    p_status.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p_status.add_argument("--json", action="store_true")
    p_status.set_defaults(func=cmd_status)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
