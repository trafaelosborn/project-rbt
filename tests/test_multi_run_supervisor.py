"""Tests for src.control.multi_run_supervisor."""

import json
from pathlib import Path

from src.control import multi_run_supervisor as supervisor


def _write_corpus(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"sequences": [["a", "b"]]}), encoding="utf-8")


def test_build_jobs_expands_language_seed_matrix_and_uses_repo_corpora(tmp_path):
    jobs = supervisor.build_jobs(
        languages=["french", "portuguese"],
        seeds=[42, 43],
        output_root=tmp_path,
        total_target_proposals=1000,
        live_event_mode="accepted_only",
    )

    assert [job.job_id for job in jobs] == [
        "french_seed_42",
        "french_seed_43",
        "portuguese_seed_42",
        "portuguese_seed_43",
    ]
    assert jobs[0].start_corpus.name == "french_tokens.json"
    assert "processed" in str(jobs[0].start_corpus)
    assert jobs[2].start_corpus.name == "portuguese_tokens.json"
    assert "sequestered" in str(jobs[2].start_corpus)
    assert jobs[0].output_dir == tmp_path / "french" / "seed_42"


def test_job_command_launches_long_run_v5_with_resource_light_logging(tmp_path):
    corpus = tmp_path / "french_tokens.json"
    _write_corpus(corpus)
    job = supervisor.SupervisorJob(
        language="french",
        seed=42,
        start_corpus=corpus,
        output_dir=tmp_path / "run",
        total_target_proposals=1000,
        candidate_count=16,
        live_event_mode="accepted_only",
        features_freeze_commit="abc123",
        features_freeze_sha256="b" * 64,
        corpus_budget=20_000,
    )

    cmd = job.command()

    assert cmd[:3] == [supervisor.sys.executable, "-m", "src.retrodiction.long_run_v5"]
    assert "--language" in cmd
    assert "french" in cmd
    assert "--target-proposals" in cmd
    assert "1000" in cmd
    assert "--use-fortran-batch" in cmd
    assert "--live-event-mode" in cmd
    assert "accepted_only" in cmd
    assert "--features-freeze-commit" in cmd
    assert "abc123" in cmd
    assert "--features-freeze-sha256" in cmd
    assert "b" * 64 in cmd
    assert "--corpus-budget" in cmd
    assert "20000" in cmd


def test_write_and_load_supervisor_manifest_roundtrip(tmp_path):
    corpus = tmp_path / "french_tokens.json"
    _write_corpus(corpus)
    job = supervisor.SupervisorJob(
        language="french",
        seed=42,
        start_corpus=corpus,
        output_dir=tmp_path / "french" / "seed_42",
        total_target_proposals=1000,
    )

    path = supervisor.write_supervisor_manifest(
        tmp_path,
        jobs=[job],
        profile=supervisor.PROFILES["polite"],
        max_parallel=3,
    )
    payload, loaded_jobs = supervisor.load_supervisor_manifest(tmp_path)

    assert path.exists()
    assert payload["job_count"] == 1
    assert payload["max_parallel"] == 3
    assert loaded_jobs == [job]


def test_pause_jobs_writes_global_and_per_job_sentinels(tmp_path):
    corpus = tmp_path / "french_tokens.json"
    _write_corpus(corpus)
    jobs = [
        supervisor.SupervisorJob(
            language="french",
            seed=42,
            start_corpus=corpus,
            output_dir=tmp_path / "french" / "seed_42",
            total_target_proposals=1000,
        ),
        supervisor.SupervisorJob(
            language="spanish",
            seed=42,
            start_corpus=corpus,
            output_dir=tmp_path / "spanish" / "seed_42",
            total_target_proposals=1000,
        ),
    ]
    supervisor.write_supervisor_manifest(
        tmp_path,
        jobs=jobs,
        profile=supervisor.PROFILES["polite"],
        max_parallel=2,
    )

    sentinels = supervisor.pause_jobs(tmp_path)

    assert (tmp_path / supervisor.PAUSE_SENTINEL).exists()
    assert sentinels == [job.stop_sentinel for job in jobs]
    assert all(path.exists() for path in sentinels)

    removed = supervisor.clear_pause_requests(tmp_path)
    assert removed == 3
    assert not (tmp_path / supervisor.PAUSE_SENTINEL).exists()
    assert not any(path.exists() for path in sentinels)


def test_read_job_status_from_child_manifest(tmp_path):
    corpus = tmp_path / "french_tokens.json"
    _write_corpus(corpus)
    job = supervisor.SupervisorJob(
        language="french",
        seed=42,
        start_corpus=corpus,
        output_dir=tmp_path / "french" / "seed_42",
        total_target_proposals=1000,
    )
    job.output_dir.mkdir(parents=True)
    (job.output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "status": "running",
                "cumulative_proposals": 2000,
                "current_corpus": "current.json",
                "updated_utc": "2026-06-11T00:00:00+00:00",
                "blocks": [
                    {
                        "final_latin_structural_score": -1.2,
                        "final_latin_form_score": 0.7,
                        "final_family_alignment_score": 0.4,
                    },
                    {
                        "final_latin_structural_score": -1.1,
                        "final_latin_form_score": 0.72,
                        "final_family_alignment_score": 0.43,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    status = supervisor.read_job_status(job)

    assert status.status == "running"
    assert status.cumulative_proposals == 2000
    assert status.blocks_completed == 2
    assert status.last_struct == -1.1
    assert status.last_form == 0.72
    assert status.last_align == 0.43


def test_should_launch_skips_success_and_running_by_default():
    base = supervisor.JobStatus(
        job_id="french_seed_42",
        language="french",
        seed=42,
        output_dir=Path("out"),
        status="complete",
        cumulative_proposals=1000,
        blocks_completed=1,
        last_struct=None,
        last_form=None,
        last_align=None,
        current_corpus=None,
        last_error=None,
    )
    running = supervisor.JobStatus(
        **{**base.to_dict(), "status": "running", "output_dir": Path("out")}
    )
    failed = supervisor.JobStatus(
        **{**base.to_dict(), "status": "failed", "output_dir": Path("out")}
    )

    assert supervisor.should_launch(base) is False
    assert supervisor.should_launch(running) is False
    assert supervisor.should_launch(running, resume_running=True) is True
    assert supervisor.should_launch(failed) is False
    assert supervisor.should_launch(failed, retry_failed=True) is True


def test_run_supervisor_dry_run_writes_manifest_without_launching(tmp_path):
    corpus = tmp_path / "french_tokens.json"
    _write_corpus(corpus)
    job = supervisor.SupervisorJob(
        language="french",
        seed=42,
        start_corpus=corpus,
        output_dir=tmp_path / "french" / "seed_42",
        total_target_proposals=1000,
    )

    summary = supervisor.run_supervisor(
        output_root=tmp_path,
        jobs=[job],
        profile=supervisor.PROFILES["polite"],
        dry_run=True,
    )

    assert (tmp_path / supervisor.SUPERVISOR_MANIFEST).exists()
    assert summary["job_count"] == 1
    assert summary["counts"] == {"not_found": 1}
