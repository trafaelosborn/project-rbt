"""Tests for src.retrodiction.long_run_v4."""

import json
import shutil
from pathlib import Path

import pytest

from src.retrodiction import long_run_v4
from src.retrodiction.long_run_v4 import LongRunConfig, run_long_continuation


def _write_corpus(path: Path, sequences: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump({"sequences": sequences}, fh, ensure_ascii=False, indent=2)


class _FakeBlockEngine:
    def __init__(
        self,
        language,
        source_sequences,
        latin_structural_ref,
        latin_form_ref,
        config,
        output_dir,
        references,
    ):
        self.language = language
        self.source_sequences = source_sequences
        self.config = config
        self.output_dir = Path(output_dir)

    def run(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        corpora_dir = self.output_dir / "corpora"
        previews_dir = self.output_dir / "previews"
        corpora_dir.mkdir(parents=True, exist_ok=True)
        previews_dir.mkdir(parents=True, exist_ok=True)

        block_tag = self.output_dir.name
        next_sequences = [list(seq) for seq in self.source_sequences]
        next_sequences.append([block_tag, "latinish"])

        corpus_path = corpora_dir / f"{block_tag}_tokens.json"
        preview_path = previews_dir / f"{block_tag}_preview.txt"
        _write_corpus(corpus_path, next_sequences)
        preview_path.write_text("preview\n", encoding="utf-8")

        proposals = int(self.config.max_proposals)
        summary = {
            "config": {
                "seed": 42,
            },
            "proposals_attempted": proposals,
            "accepted_mutation_stages": min(1, proposals),
            "halt_reason": "max_proposals",
            "best_stage_id": block_tag.upper(),
            "best_corpus_json": str(corpus_path),
            "best_preview_txt": str(preview_path),
            "final_latin_structural_score": -1.5 + 0.01 * proposals,
            "final_latin_form_score": 0.4 + 0.001 * proposals,
            "final_family_alignment_score": 0.3 + 0.001 * proposals,
            "final_coherence_label": "coherent",
        }
        with (self.output_dir / "run_summary.json").open("w", encoding="utf-8") as fh:
            json.dump(summary, fh, ensure_ascii=False, indent=2)
        return []


class _LatinHitFakeEngine(_FakeBlockEngine):
    def run(self):
        super().run()
        summary_path = self.output_dir / "run_summary.json"
        with summary_path.open(encoding="utf-8") as fh:
            summary = json.load(fh)
        summary["final_latin_structural_score"] = 0.0
        summary["final_latin_form_score"] = 1.0
        summary["final_family_alignment_score"] = 1.0
        with summary_path.open("w", encoding="utf-8") as fh:
            json.dump(summary, fh, ensure_ascii=False, indent=2)
        return []


class _FailingFakeEngine(_FakeBlockEngine):
    def run(self):
        raise RuntimeError("boom")


class TestLongRunV4:
    def test_run_long_continuation_writes_multi_block_manifest(self):
        root = (Path.cwd() / "data" / "retrodiction" / "_test_long_run_v4_multi").resolve()
        start_corpus = root / "seed" / "seed_tokens.json"
        try:
            shutil.rmtree(root, ignore_errors=True)
            _write_corpus(start_corpus, [["de", "la"], ["terre"]])

            cfg = LongRunConfig(
                language="french",
                start_corpus=start_corpus,
                output_dir=root / "run",
                total_target_proposals=10,
                block_proposals=4,
            )
            manifest = run_long_continuation(cfg, engine_class=_FakeBlockEngine)

            assert manifest["status"] == "complete"
            assert manifest["completed_block_proposals"] == 10
            assert manifest["cumulative_proposals"] == 10
            assert len(manifest["blocks"]) == 3
            assert manifest["seed_audit"] == {
                "requested_seed": 42,
                "engine_seed": 42,
                "seeds_match": True,
            }
            assert manifest["blocks"][0]["started_from_corpus"] == str(start_corpus)
            assert manifest["blocks"][-1]["proposals_attempted"] == 2
            assert manifest["blocks"][0]["seed_audit"] == {
                "requested_seed": 42,
                "engine_seed": 42,
                "seeds_match": True,
            }
            assert Path(manifest["current_corpus"]).exists()
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_run_long_continuation_normalizes_block_summary_seed_and_records_seed_audit(self):
        root = (Path.cwd() / "data" / "retrodiction" / "_test_long_run_v4_seed_audit").resolve()
        start_corpus = root / "seed" / "seed_tokens.json"
        try:
            shutil.rmtree(root, ignore_errors=True)
            _write_corpus(start_corpus, [["de", "la"], ["terre"]])

            cfg = LongRunConfig(
                language="french",
                start_corpus=start_corpus,
                output_dir=root / "run",
                total_target_proposals=4,
                block_proposals=4,
                seed=45,
            )
            manifest = run_long_continuation(cfg, engine_class=_FakeBlockEngine)

            assert manifest["seed_audit"] == {
                "requested_seed": 45,
                "engine_seed": 45,
                "seeds_match": True,
            }
            assert manifest["blocks"][0]["seed_audit"] == {
                "requested_seed": 45,
                "engine_seed": 45,
                "seeds_match": True,
            }

            summary_path = Path(manifest["blocks"][0]["summary_path"])
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            assert summary["config"]["seed"] == 45
            assert summary["config"]["requested_seed"] == 45
            assert summary["seed_audit"] == {
                "requested_seed": 45,
                "engine_seed": 45,
                "seeds_match": True,
            }
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_run_long_continuation_resumes_from_existing_manifest(self):
        root = (Path.cwd() / "data" / "retrodiction" / "_test_long_run_v4_resume").resolve()
        start_corpus = root / "seed" / "seed_tokens.json"
        resumed_corpus = root / "resume" / "resume_tokens.json"
        manifest_path = root / "run" / "manifest.json"
        try:
            shutil.rmtree(root, ignore_errors=True)
            _write_corpus(start_corpus, [["de", "la"]])
            _write_corpus(resumed_corpus, [["resume", "corpus"]])

            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest = {
                "created_utc": "2026-04-08T00:00:00+00:00",
                "updated_utc": "2026-04-08T00:00:00+00:00",
                "status": "running",
                "config": {},
                "starting_proposals": 5,
                "completed_block_proposals": 4,
                "cumulative_proposals": 9,
                "current_corpus": str(resumed_corpus),
                "latin_hit": False,
                "blocks": [
                    {
                        "block": "block_0001",
                        "started_from_corpus": str(start_corpus),
                        "ended_at_corpus": str(resumed_corpus),
                    }
                ],
            }
            with manifest_path.open("w", encoding="utf-8") as fh:
                json.dump(manifest, fh, ensure_ascii=False, indent=2)

            cfg = LongRunConfig(
                language="french",
                start_corpus=start_corpus,
                output_dir=manifest_path.parent,
                total_target_proposals=12,
                block_proposals=4,
                starting_proposals=5,
            )
            result = run_long_continuation(cfg, engine_class=_FakeBlockEngine)

            assert result["status"] == "complete"
            assert result["completed_block_proposals"] == 7
            assert result["cumulative_proposals"] == 12
            assert len(result["blocks"]) == 2
            assert result["blocks"][-1]["started_from_corpus"] == str(resumed_corpus)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_run_long_continuation_stops_when_latin_targets_are_hit(self):
        root = (Path.cwd() / "data" / "retrodiction" / "_test_long_run_v4_latin_hit").resolve()
        start_corpus = root / "seed" / "seed_tokens.json"
        try:
            shutil.rmtree(root, ignore_errors=True)
            _write_corpus(start_corpus, [["de", "la"], ["terre"]])

            cfg = LongRunConfig(
                language="french",
                start_corpus=start_corpus,
                output_dir=root / "run",
                total_target_proposals=20,
                block_proposals=5,
            )
            manifest = run_long_continuation(cfg, engine_class=_LatinHitFakeEngine)

            assert manifest["status"] == "joint_hit"
            assert manifest["latin_hit"] is True
            assert len(manifest["blocks"]) == 1
            assert manifest["completed_block_proposals"] == 5
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_run_long_continuation_persists_failed_manifest_before_first_block_completes(self):
        root = (Path.cwd() / "data" / "retrodiction" / "_test_long_run_v4_failed_startup").resolve()
        start_corpus = root / "seed" / "seed_tokens.json"
        manifest_path = root / "run" / "manifest.json"
        try:
            shutil.rmtree(root, ignore_errors=True)
            _write_corpus(start_corpus, [["de", "la"], ["terre"]])

            cfg = LongRunConfig(
                language="french",
                start_corpus=start_corpus,
                output_dir=root / "run",
                total_target_proposals=20,
                block_proposals=5,
            )
            with pytest.raises(RuntimeError, match="boom"):
                run_long_continuation(cfg, engine_class=_FailingFakeEngine)

            assert manifest_path.exists()
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            assert manifest["status"] == "failed"
            assert manifest["blocks"] == []
            assert manifest["last_error"] == "RuntimeError: boom"
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_run_long_continuation_can_write_block_validator_snapshots(self, monkeypatch):
        root = (Path.cwd() / "data" / "retrodiction" / "_test_long_run_v4_validator_snapshots").resolve()
        start_corpus = root / "seed" / "seed_tokens.json"
        calls: list[dict] = []

        def _fake_compare(run_manifest_path, output_prefix=None, validator_ids=None, output_dir=None, block_ids=None):
            calls.append(
                {
                    "run_manifest_path": str(run_manifest_path),
                    "output_prefix": output_prefix,
                    "validator_ids": validator_ids,
                    "output_dir": str(output_dir),
                    "block_ids": block_ids,
                }
            )
            output_dir.mkdir(parents=True, exist_ok=True)
            csv_path = output_dir / f"{output_prefix}_vs_validator_bank.csv"
            json_path = output_dir / f"{output_prefix}_vs_validator_bank.json"
            summary_path = output_dir / f"{output_prefix}_vs_validator_bank_chronology.json"
            csv_path.write_text("ok\n", encoding="utf-8")
            json_path.write_text("{}", encoding="utf-8")
            summary_path.write_text("{}", encoding="utf-8")
            return {
                "csv_path": str(csv_path),
                "json_path": str(json_path),
                "summary_path": str(summary_path),
                "validator_count": 2,
                "block_count": 1,
                "summary": {},
            }

        try:
            shutil.rmtree(root, ignore_errors=True)
            _write_corpus(start_corpus, [["de", "la"], ["terre"]])
            monkeypatch.setattr(long_run_v4, "compare_run_manifest_to_validator_bank", _fake_compare)

            cfg = LongRunConfig(
                language="french",
                start_corpus=start_corpus,
                output_dir=root / "run",
                total_target_proposals=8,
                block_proposals=4,
                validator_set=["old_french", "middle_french"],
                validator_snapshot_every_blocks=1,
            )
            manifest = run_long_continuation(cfg, engine_class=_FakeBlockEngine)

            assert manifest["status"] == "complete"
            assert len(calls) == 2
            assert calls[0]["validator_ids"] == ["old_french", "middle_french"]
            assert calls[0]["block_ids"] == ["block_0001"]
            assert "validator_snapshot" in manifest["blocks"][0]
            assert Path(manifest["blocks"][0]["validator_snapshot"]["summary_path"]).exists()
        finally:
            shutil.rmtree(root, ignore_errors=True)
