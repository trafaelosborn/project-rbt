"""Tests for src.ingest.historical."""

import json
import shutil
from pathlib import Path

from src.ingest.historical import ingest


class TestHistoricalIngest:
    def test_ingest_writes_tokens_manifest_and_fingerprints(self):
        root = (Path.cwd() / "project_rbt" / "data" / "_test_historical_ingest").resolve()
        try:
            shutil.rmtree(root, ignore_errors=True)
            input_dir = root / "raw" / "historical" / "old_french"
            output_dir = root / "processed" / "historical"
            matrices_dir = root / "matrices"
            input_dir.mkdir(parents=True, exist_ok=True)

            (input_dir / "sample_a.txt").write_text(
                "Li rois parla. La dame respondi.",
                encoding="utf-8",
            )
            (input_dir / "sample_b.md").write_text(
                "Uns chevaliers vint.\nPuis il canta.",
                encoding="utf-8",
            )
            (input_dir / "README.md").write_text(
                "# metadata only",
                encoding="utf-8",
            )
            (input_dir / "_scratch.txt").write_text(
                "should not be ingested",
                encoding="utf-8",
            )

            tokens_path = ingest(
                name="old_french",
                language="french",
                period_label="Old French",
                input_dir=input_dir,
                output_dir=output_dir,
                matrices_dir=matrices_dir,
                source="test_fixture",
            )

            assert tokens_path.exists()
            assert (output_dir / "old_french_manifest.json").exists()
            assert (matrices_dir / "old_french_cooccurrence.npy").exists()
            assert (matrices_dir / "old_french_positional.npy").exists()
            assert (matrices_dir / "old_french_ngram_meta.json").exists()

            with tokens_path.open(encoding="utf-8") as fh:
                corpus = json.load(fh)

            assert corpus["language"] == "old_french"
            assert corpus["branch_language"] == "french"
            assert corpus["historical_period"] == "Old French"
            assert corpus["source"] == "test_fixture"
            assert corpus["file_count"] == 2
            assert corpus["sequence_count"] > 0
            assert corpus["total_tokens"] > 0
        finally:
            shutil.rmtree(root, ignore_errors=True)
