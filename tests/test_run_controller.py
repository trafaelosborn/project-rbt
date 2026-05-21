"""Tests for src.control.run_controller."""

import json
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.control.run_config import RunConfig
from src.control.run_controller import RunController, RunStatus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_corpus(tmp_path: Path) -> Path:
    p = tmp_path / "corpus_tokens.json"
    p.write_text(json.dumps({"sequences": [["a", "b"], ["c", "d"]]}), encoding="utf-8")
    return p


def _make_manifest(tmp_path: Path, status: str = "complete", blocks: int = 3) -> Path:
    block_list = [
        {
            "block": f"block_{i+1:04d}",
            "final_latin_structural_score": -0.5 + i * 0.1,
            "final_latin_form_score": 0.6 + i * 0.05,
            "final_family_alignment_score": 0.55,
            "proposals_attempted": 1000,
            "best_corpus_json": str(tmp_path / f"corpus_{i}.json"),
            "ended_at_corpus": str(tmp_path / f"corpus_{i}.json"),
        }
        for i in range(blocks)
    ]
    manifest = {
        "status": status,
        "cumulative_proposals": blocks * 1000,
        "blocks": block_list,
        "current_corpus": str(tmp_path / f"corpus_{blocks - 1}.json"),
        "updated_utc": "2026-04-12T00:00:00+00:00",
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def _valid_config(tmp_path: Path) -> RunConfig:
    corpus = _make_corpus(tmp_path)
    return RunConfig(
        source_language="french",
        target_language="latin",
        start_corpus=corpus,
        until_mode="latin_hit",
        candidate_count=8,
        block_proposals=1000,
    )


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

class TestStatus:
    def test_status_not_found_when_no_manifest(self, tmp_path):
        controller = RunController()
        status = controller.status(tmp_path / "nonexistent")
        assert status.status == "not_found"
        assert status.cumulative_proposals == 0

    def test_status_reads_manifest(self, tmp_path):
        _make_manifest(tmp_path, status="running", blocks=5)
        controller = RunController()
        status = controller.status(tmp_path)
        assert status.status == "running"
        assert status.cumulative_proposals == 5000
        assert status.blocks_completed == 5

    def test_status_extracts_scores(self, tmp_path):
        _make_manifest(tmp_path, blocks=3)
        controller = RunController()
        status = controller.status(tmp_path)
        assert status.last_struct is not None
        assert status.last_form is not None

    def test_status_is_terminal_for_complete(self, tmp_path):
        _make_manifest(tmp_path, status="complete")
        controller = RunController()
        status = controller.status(tmp_path)
        assert status.is_terminal

    def test_status_reads_failed_manifest_error(self, tmp_path):
        manifest_path = _make_manifest(tmp_path, status="failed", blocks=0)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["last_error"] = "RuntimeError: boom"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        controller = RunController()
        status = controller.status(tmp_path)
        assert status.status == "failed"
        assert status.is_terminal
        assert status.last_error == "RuntimeError: boom"

    def test_status_is_not_terminal_for_running(self, tmp_path):
        _make_manifest(tmp_path, status="running")
        controller = RunController()
        status = controller.status(tmp_path)
        assert not status.is_terminal

    def test_summary_line_format(self, tmp_path):
        _make_manifest(tmp_path, status="complete", blocks=2)
        controller = RunController()
        line = controller.status(tmp_path).summary_line()
        assert "complete" in line
        assert "proposals=" in line
        assert "struct=" in line


# ---------------------------------------------------------------------------
# Stop sentinel
# ---------------------------------------------------------------------------

class TestStopRun:
    def test_stop_run_writes_sentinel(self, tmp_path):
        controller = RunController()
        run_dir = tmp_path / "myrun"
        run_dir.mkdir()
        controller.stop_run(run_dir)
        assert (run_dir / "stop_requested").exists()

    def test_stop_run_creates_parent_dir(self, tmp_path):
        controller = RunController()
        run_dir = tmp_path / "deep" / "nested" / "run"
        controller.stop_run(run_dir)
        assert (run_dir / "stop_requested").exists()


# ---------------------------------------------------------------------------
# Launch validation guard
# ---------------------------------------------------------------------------

class TestLaunchValidation:
    def test_launch_rejects_invalid_config(self, tmp_path):
        cfg = _valid_config(tmp_path)
        cfg.candidate_count = 7   # not a preset
        controller = RunController()
        with pytest.raises(Exception):
            controller.launch_run(cfg, tmp_path / "out")

    def test_launch_locks_config(self, tmp_path):
        """launch_run must lock the config before calling the driver."""
        corpus = _make_corpus(tmp_path)
        cfg = RunConfig(
            source_language="french",
            target_language="latin",
            start_corpus=corpus,
            until_mode="latin_hit",
            candidate_count=8,
            block_proposals=1000,
        )
        locked_at_call = {}

        def fake_driver(long_cfg):
            locked_at_call["locked"] = cfg.is_locked
            out = Path(long_cfg.output_dir)
            out.mkdir(parents=True, exist_ok=True)
            (out / "manifest.json").write_text(
                json.dumps({
                    "status": "complete",
                    "cumulative_proposals": 0,
                    "blocks": [],
                    "current_corpus": str(corpus),
                    "updated_utc": "2026-04-12T00:00:00+00:00",
                }),
                encoding="utf-8",
            )

        with patch("src.control._driver_adapter.run_long_continuation", new=fake_driver):
            controller = RunController()
            controller.launch_run(cfg, tmp_path / "out")

        assert locked_at_call.get("locked") is True


# ---------------------------------------------------------------------------
# Chain run
# ---------------------------------------------------------------------------

class TestChainRun:
    def test_chain_overrides_start_corpus(self, tmp_path):
        manifest_dir = tmp_path / "prev_run"
        manifest_dir.mkdir()
        best_corpus = tmp_path / "best_corpus.json"
        best_corpus.write_text('{"sequences": []}', encoding="utf-8")
        manifest = {
            "status": "complete",
            "current_corpus": str(best_corpus),
            "cumulative_proposals": 1000,
            "blocks": [],
            "updated_utc": "2026-04-12T00:00:00+00:00",
        }
        (manifest_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

        cfg = _valid_config(tmp_path)
        original_corpus = cfg.start_corpus
        observed = {}

        def fake_driver(long_cfg):
            observed["start_corpus"] = str(long_cfg.start_corpus)
            out = Path(long_cfg.output_dir)
            out.mkdir(parents=True, exist_ok=True)
            (out / "manifest.json").write_text(
                json.dumps({
                    "status": "complete",
                    "cumulative_proposals": 0,
                    "blocks": [],
                    "current_corpus": str(best_corpus),
                    "updated_utc": "2026-04-12T00:00:00+00:00",
                }),
                encoding="utf-8",
            )

        with patch("src.control._driver_adapter.run_long_continuation", new=fake_driver):
            controller = RunController()
            controller.chain_run(
                from_manifest=manifest_dir / "manifest.json",
                config=cfg,
                output_dir=tmp_path / "new_run",
            )

        assert observed["start_corpus"] == str(best_corpus)

    def test_chain_raises_on_missing_manifest(self, tmp_path):
        cfg = _valid_config(tmp_path)
        controller = RunController()
        with pytest.raises(FileNotFoundError):
            controller.chain_run(
                from_manifest=tmp_path / "noexist" / "manifest.json",
                config=cfg,
                output_dir=tmp_path / "out",
            )


# ---------------------------------------------------------------------------
# Validate run
# ---------------------------------------------------------------------------

class TestValidateRun:
    def test_validate_run_uses_validator_bank_compare(self, tmp_path):
        _make_manifest(tmp_path, status="plateau_hit", blocks=3)
        controller = RunController()

        expected = {
            "csv_path": str(tmp_path / "out.csv"),
            "json_path": str(tmp_path / "out.json"),
            "summary_path": str(tmp_path / "out_summary.json"),
            "validator_count": 5,
            "block_count": 3,
            "summary": {},
        }

        with patch(
            "src.validation.validator_bank_compare.compare_run_manifest_to_validator_bank",
            return_value=expected,
        ) as mock_compare:
            result = controller.validate_run(tmp_path)

        assert result == expected
        mock_compare.assert_called_once_with(
            run_manifest_path=tmp_path / "manifest.json",
            output_prefix=None,
            output_dir=None,
        )

    def test_validate_run_maps_file_like_output_path_to_prefix_and_dir(self, tmp_path):
        _make_manifest(tmp_path, status="plateau_hit", blocks=2)
        controller = RunController()
        output_path = tmp_path / "validation" / "paper_run.json"

        with patch(
            "src.validation.validator_bank_compare.compare_run_manifest_to_validator_bank",
            return_value={},
        ) as mock_compare:
            controller.validate_run(tmp_path, output_path=output_path)

        mock_compare.assert_called_once_with(
            run_manifest_path=tmp_path / "manifest.json",
            output_prefix="paper_run",
            output_dir=output_path.parent,
        )
