"""Tests for src.validation.checkpoint_compare."""

import json
import shutil
from pathlib import Path

from src.fingerprint import ngram
from src.ingest.historical import ingest
from src.validation.checkpoint_compare import (
    compare_run_to_validator,
    default_checkpoint_stage_ids,
)


def _write_stage(run_dir: Path, stage_id: str, sequences: list[list[str]]) -> tuple[Path, Path]:
    corpora_dir = run_dir / "corpora"
    previews_dir = run_dir / "previews"
    matrices_dir = run_dir / "matrices"
    corpora_dir.mkdir(parents=True, exist_ok=True)
    previews_dir.mkdir(parents=True, exist_ok=True)
    matrices_dir.mkdir(parents=True, exist_ok=True)

    corpus_path = corpora_dir / f"{stage_id}_tokens.json"
    preview_path = previews_dir / f"{stage_id}_preview.txt"

    with corpus_path.open("w", encoding="utf-8") as fh:
        json.dump({"stage_id": stage_id, "sequences": sequences}, fh, ensure_ascii=False, indent=2)
    with preview_path.open("w", encoding="utf-8") as fh:
        fh.write("\n".join(" ".join(seq) for seq in sequences[:5]))

    ngram.run_from_sequences(stage_id, sequences, output_dir=matrices_dir)
    return corpus_path, preview_path


class TestCheckpointCompare:
    def test_default_checkpoint_stage_ids_uses_quartile_ladder(self):
        summary = {"stages": [{"stage_id": f"FR_v2_{i:03d}"} for i in range(62)]}
        assert default_checkpoint_stage_ids(summary) == [
            "FR_v2_000",
            "FR_v2_015",
            "FR_v2_030",
            "FR_v2_045",
            "FR_v2_058",
            "FR_v2_061",
        ]

    def test_compare_run_to_validator_picks_matching_stage(self):
        root = (Path.cwd() / "project_rbt" / "data" / "_test_checkpoint_compare").resolve()
        try:
            shutil.rmtree(root, ignore_errors=True)
            run_dir = root / "retrodiction" / "french" / "v2_convergence"
            raw_hist_dir = root / "raw" / "historical" / "old_french"
            processed_hist_dir = root / "processed" / "historical"
            matrices_dir = root / "matrices"
            output_path = root / "validation" / "comparison.json"

            raw_hist_dir.mkdir(parents=True, exist_ok=True)
            validator_text = "Li rois parla. La dame respondi. Li rois parla. La dame respondi."
            (raw_hist_dir / "validator.txt").write_text(validator_text, encoding="utf-8")

            validator_tokens_path = ingest(
                name="old_french",
                language="french",
                period_label="Old French",
                input_dir=raw_hist_dir,
                output_dir=processed_hist_dir,
                matrices_dir=matrices_dir,
                source="test_fixture",
            )

            stage_0 = [[
                "de",
                "la",
                "maison",
                "de",
                "la",
                "ville",
                "de",
                "la",
                "cour",
                "et",
                "le",
                "roi",
            ]] * 6
            stage_1 = [["li", "rois", "parla"], ["la", "dame", "respondi"]] * 6
            stage_2 = [[
                "unum",
                "mare",
                "cantum",
                "porta",
                "forum",
                "militum",
                "scriptura",
                "legatum",
                "bellorum",
            ]] * 6

            stage_specs = [
                ("FR_v2_000", stage_0, "seed"),
                ("FR_v2_001", stage_1, "token_char_edit"),
                ("FR_v2_002", stage_2, "suffix_family_rewrite"),
            ]

            stages = []
            for i, (stage_id, sequences, operator) in enumerate(stage_specs):
                corpus_path, preview_path = _write_stage(run_dir, stage_id, sequences)
                stages.append(
                    {
                        "stage_id": stage_id,
                        "iteration": i,
                        "mutation_operator": operator,
                        "artifacts": {
                            "corpus_json": str(corpus_path),
                            "preview_txt": str(preview_path),
                        },
                        "latin_structural_score": -1.0 + i * 0.1,
                        "latin_form_score": 0.2 + i * 0.1,
                        "total_score": -0.8 + i * 0.1,
                        "diagnostics": {"coherence_label": "coherent"},
                    }
                )

            run_summary_path = run_dir / "run_summary.json"
            run_dir.mkdir(parents=True, exist_ok=True)
            with run_summary_path.open("w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "language": "french",
                        "algorithm": "relational_v2",
                        "stages": stages,
                    },
                    fh,
                    ensure_ascii=False,
                    indent=2,
                )

            result = compare_run_to_validator(
                run_summary_path=run_summary_path,
                validator_tokens_path=validator_tokens_path,
                checkpoint_stage_ids=["FR_v2_000", "FR_v2_001", "FR_v2_002"],
                output_path=output_path,
            )

            assert output_path.exists()
            assert result["best_by_structural_distance"]["stage_id"] == "FR_v2_001"
            assert result["best_by_form_score"]["stage_id"] == "FR_v2_001"
        finally:
            shutil.rmtree(root, ignore_errors=True)
