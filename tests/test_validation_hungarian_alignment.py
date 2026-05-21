"""Tests for src.validation.hungarian_alignment."""

import json
import shutil
from pathlib import Path

from src.validation.hungarian_alignment import (
    FamilyAlignmentConfig,
    compare_run_to_reference,
    extract_family_inventory,
    hungarian_alignment_diagnostics,
)


def _write_corpus(path: Path, sequences: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump({"sequences": sequences}, fh, ensure_ascii=False, indent=2)


class TestHungarianAlignment:
    def test_extract_family_inventory_collects_multiple_family_kinds(self):
        cfg = FamilyAlignmentConfig(
            max_suffix_families=4,
            max_prefix_families=4,
            max_short_token_families=4,
            min_short_token_count=1,
        )
        sequences = [
            ["de", "la", "maison", "nation", "ration", "conduire", "conference"],
            ["de", "la", "raison", "vision", "convoi", "combat", "compagnie"],
        ] * 4

        inventory = extract_family_inventory("toy", sequences, cfg)
        kinds = {fam.kind for fam in inventory.families}
        assert "suffix" in kinds
        assert "prefix" in kinds
        assert "short_token" in kinds

    def test_alignment_prefers_more_similar_inventory(self):
        cfg = FamilyAlignmentConfig(
            max_suffix_families=6,
            max_prefix_families=6,
            max_short_token_families=6,
            min_short_token_count=1,
        )
        reference_sequences = [
            ["de", "la", "forum", "bellum", "datum", "romanum", "comitium"],
            ["de", "la", "ludum", "datum", "bellum", "consilium", "forum"],
        ] * 5
        similar_sequences = [
            ["de", "la", "bellum", "datum", "forum", "comitium"],
            ["de", "la", "ludum", "forum", "datum", "romanum"],
        ] * 4
        dissimilar_sequences = [
            ["les", "maisons", "bleues", "chantent", "vite"],
            ["ainsi", "font", "les", "choses", "modernes"],
        ] * 4

        ref = extract_family_inventory("latin_like", reference_sequences, cfg)
        similar = extract_family_inventory("similar", similar_sequences, cfg)
        dissimilar = extract_family_inventory("dissimilar", dissimilar_sequences, cfg)

        similar_diag = hungarian_alignment_diagnostics(similar, ref, cfg)
        dissimilar_diag = hungarian_alignment_diagnostics(dissimilar, ref, cfg)

        assert similar_diag["family_alignment_score"] > dissimilar_diag["family_alignment_score"]
        assert similar_diag["family_alignment_cost"] < dissimilar_diag["family_alignment_cost"]

    def test_compare_run_to_reference_picks_best_aligned_stage(self):
        root = (Path.cwd() / "project_rbt" / "data" / "_test_hungarian_alignment").resolve()
        try:
            shutil.rmtree(root, ignore_errors=True)
            cfg = FamilyAlignmentConfig(
                max_suffix_families=6,
                max_prefix_families=6,
                max_short_token_families=6,
                min_short_token_count=1,
            )

            reference_sequences = [
                ["de", "la", "forum", "bellum", "datum", "romanum", "consilium", "conventum"],
                ["de", "la", "datum", "forum", "consilium", "bellum", "conlatum", "romanum"],
            ] * 5
            reference_inventory = extract_family_inventory("latin_like", reference_sequences, cfg)

            run_dir = root / "retrodiction" / "french" / "toy_run"
            corpora_dir = run_dir / "corpora"
            previews_dir = run_dir / "previews"
            output_path = root / "validation" / "alignment.json"

            stage_0 = [["de", "la", "maison", "nation", "vision", "compagnie"]] * 6
            stage_1 = [
                ["de", "la", "forum", "datum", "bellum", "consilium", "conlatum", "romanum"],
                ["de", "la", "ludum", "datum", "forum", "conventum", "bellum", "romanum"],
            ] * 3
            stage_2 = [["les", "bleues", "choses", "modernes", "chantent"]] * 6

            stages = []
            for i, (stage_id, sequences, operator) in enumerate(
                [
                    ("FR_v4_000", stage_0, "seed"),
                    ("FR_v4_001", stage_1, "macro_bundle_rewrite"),
                    ("FR_v4_002", stage_2, "function_word_burst"),
                ]
            ):
                corpus_path = corpora_dir / f"{stage_id}_tokens.json"
                preview_path = previews_dir / f"{stage_id}_preview.txt"
                _write_corpus(corpus_path, sequences)
                preview_path.parent.mkdir(parents=True, exist_ok=True)
                preview_path.write_text("\n".join(" ".join(seq) for seq in sequences[:2]), encoding="utf-8")
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
                    {"language": "french", "algorithm": "relational_v4", "stages": stages},
                    fh,
                    ensure_ascii=False,
                    indent=2,
                )

            result = compare_run_to_reference(
                run_summary_path=run_summary_path,
                reference_inventory=reference_inventory,
                checkpoint_stage_ids=["FR_v4_000", "FR_v4_001", "FR_v4_002"],
                output_path=output_path,
                config=cfg,
            )

            assert output_path.exists()
            assert result["best_by_family_alignment"]["stage_id"] == "FR_v4_001"
            assert result["best_by_lowest_cost"]["stage_id"] == "FR_v4_001"
        finally:
            shutil.rmtree(root, ignore_errors=True)
