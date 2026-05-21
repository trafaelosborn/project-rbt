"""
Control-Bank Comparison
=======================

Compare selected checkpoints from a completed retrodiction run against the
project's null/control bank:

- Markov noise floor
- Sumerian structured non-IE control
- Portuguese withheld positive control

This mirrors the validator-bank comparison tooling, but keeps nulls/controls
as a separate analysis layer.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from src.fingerprint.ngram import build_profile, extract_ngrams
from src.retrodiction.similarity import (
    ReferenceSet,
    cosine_similarity,
    scaled_euclidean_distance,
    structural_vector,
)
from src.sequester.guard import load_sequestered, lock_sequestration, unlock_sequestration
from src.validation.checkpoint_compare import (
    CorpusFormReference,
    _load_corpus_and_profiles,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VALIDATION_DIR = PROJECT_ROOT / "data" / "validation"


@dataclass(frozen=True)
class ControlBankRow:
    control_id: str
    label: str
    control_role: str
    source_type: str
    token_path: Path | None
    sequestered_name: str | None = None


CONTROL_BANK_ROWS: tuple[ControlBankRow, ...] = (
    ControlBankRow(
        control_id="vs_markov_noise",
        label="Markov noise",
        control_role="noise_floor",
        source_type="generated_null",
        token_path=PROJECT_ROOT / "data" / "processed" / "nulls" / "markov" / "markov_tokens.json",
    ),
    ControlBankRow(
        control_id="vs_sumerian",
        label="Sumerian",
        control_role="structured_non_ie_control",
        source_type="attested_control",
        token_path=PROJECT_ROOT / "data" / "processed" / "nulls" / "sumerian" / "sumerian_tokens.json",
    ),
    ControlBankRow(
        control_id="vs_portuguese_control",
        label="Portuguese",
        control_role="withheld_positive_control",
        source_type="sequestered_positive_control",
        token_path=None,
        sequestered_name="portuguese",
    ),
)


def _block_sort_key(block_id: str) -> int:
    try:
        return int(block_id.split("_")[-1])
    except ValueError:
        return 0


def _dedupe_path(values: list[str]) -> list[str]:
    output: list[str] = []
    for value in values:
        if not output or output[-1] != value:
            output.append(value)
    return output


def _control_rows(control_ids: list[str] | None = None) -> list[ControlBankRow]:
    rows = list(CONTROL_BANK_ROWS)
    if control_ids is None:
        return rows
    selected = set(control_ids)
    return [row for row in rows if row.control_id in selected]


def _build_profiles_from_sequences(
    sequences: list[list[str]],
) -> tuple[dict[str, float], dict[str, float]]:
    return (
        build_profile(extract_ngrams(sequences, 2), 5000),
        build_profile(extract_ngrams(sequences, 3), 5000),
    )


def _load_sequences_and_profiles(path: Path) -> tuple[list[list[str]], dict[str, float], dict[str, float]]:
    with path.open(encoding="utf-8") as fh:
        corpus = json.load(fh)

    sequences = corpus["sequences"]
    try:
        _, _, bigrams, trigrams = _load_corpus_and_profiles(path)
        return sequences, bigrams, trigrams
    except FileNotFoundError:
        return sequences, *_build_profiles_from_sequences(sequences)


def _load_sequestered_sequences_and_profiles(corpus_name: str) -> tuple[list[list[str]], dict[str, float], dict[str, float]]:
    unlock_sequestration(
        "Phase 5 validation: comparing completed bridge checkpoints against the withheld Portuguese positive control."
    )
    try:
        corpus = load_sequestered(corpus_name)
    finally:
        lock_sequestration()

    sequences = corpus["sequences"]
    return sequences, *_build_profiles_from_sequences(sequences)


def _load_control_reference(row: ControlBankRow) -> dict:
    if row.sequestered_name is not None:
        sequences, bigrams, trigrams = _load_sequestered_sequences_and_profiles(row.sequestered_name)
        tokens_path = str(PROJECT_ROOT / "data" / "sequestered" / row.sequestered_name / f"{row.sequestered_name}_tokens.json")
    else:
        if row.token_path is None or not row.token_path.exists():
            raise FileNotFoundError(f"Control corpus not found: {row.token_path}")
        sequences, bigrams, trigrams = _load_sequences_and_profiles(row.token_path)
        tokens_path = str(row.token_path)

    return {
        "bank": row,
        "vec": structural_vector(sequences, bigrams, trigrams),
        "form_ref": CorpusFormReference.from_sequences(row.label, sequences),
        "tokens_path": tokens_path,
    }


def compare_run_manifest_to_control_bank(
    run_manifest_path: Path,
    output_prefix: str | None = None,
    control_ids: list[str] | None = None,
    output_dir: Path | None = None,
    block_ids: list[str] | None = None,
) -> dict:
    with run_manifest_path.open(encoding="utf-8") as fh:
        manifest = json.load(fh)

    run_id = run_manifest_path.parent.name
    blocks = manifest.get("blocks", [])
    if block_ids is not None:
        block_filter = set(block_ids)
        blocks = [block for block in blocks if block.get("block") in block_filter]

    controls = _control_rows(control_ids)
    references = ReferenceSet()
    scale = references.real_language_scale

    control_refs = {
        row.control_id: _load_control_reference(row)
        for row in controls
    }

    rows: list[dict] = []
    for block in blocks:
        stage_id = block["best_stage_id"]
        corpus_path = Path(block["best_corpus_json"])
        sequences, bigrams, trigrams = _load_sequences_and_profiles(corpus_path)
        vec = structural_vector(sequences, bigrams, trigrams)

        for control in controls:
            control_ref = control_refs[control.control_id]
            form_scores = control_ref["form_ref"].score(sequences)
            rows.append(
                {
                    "run_id": run_id,
                    "block_id": block["block"],
                    "stage_id": stage_id,
                    "control_id": control.control_id,
                    "control_label": control.label,
                    "control_role": control.control_role,
                    "control_source_type": control.source_type,
                    "control_tokens_path": control_ref["tokens_path"],
                    "latin_structural_score": block.get("final_latin_structural_score"),
                    "latin_form_score": block.get("final_latin_form_score"),
                    "family_alignment_score": block.get("final_family_alignment_score"),
                    "coherence_label": block.get("final_coherence_label"),
                    "control_structural_cosine": cosine_similarity(vec, control_ref["vec"]),
                    "control_structural_distance": scaled_euclidean_distance(vec, control_ref["vec"], scale),
                    "control_form_score": form_scores["validator_form_score"],
                    "control_char_bigram_cosine": form_scores["validator_char_bigram_cosine"],
                    "control_char_trigram_cosine": form_scores["validator_char_trigram_cosine"],
                    "control_suffix_cosine": form_scores["validator_suffix_cosine"],
                    "corpus_json": str(corpus_path),
                    "preview_txt": block.get("best_preview_txt"),
                }
            )

    rows.sort(key=lambda item: (_block_sort_key(item["block_id"]), item["control_id"]))

    by_block: dict[str, list[dict]] = {}
    for row in rows:
        by_block.setdefault(row["block_id"], []).append(row)

    nearest_structural_by_block = []
    nearest_form_by_block = []
    for block_id in sorted(by_block, key=_block_sort_key):
        block_rows = by_block[block_id]
        nearest_structural_by_block.append(min(block_rows, key=lambda item: item["control_structural_distance"]))
        nearest_form_by_block.append(max(block_rows, key=lambda item: item["control_form_score"]))

    first_structural_win_by_control: dict[str, dict] = {}
    first_form_win_by_control: dict[str, dict] = {}
    for item in nearest_structural_by_block:
        first_structural_win_by_control.setdefault(item["control_id"], item)
    for item in nearest_form_by_block:
        first_form_win_by_control.setdefault(item["control_id"], item)

    best_structural_by_control = {}
    best_form_by_control = {}
    for control in controls:
        control_rows = [row for row in rows if row["control_id"] == control.control_id]
        best_structural_by_control[control.control_id] = min(
            control_rows, key=lambda item: item["control_structural_distance"]
        )
        best_form_by_control[control.control_id] = max(
            control_rows, key=lambda item: item["control_form_score"]
        )

    summary = {
        "structural_path": _dedupe_path([item["control_id"] for item in nearest_structural_by_block]),
        "form_path": _dedupe_path([item["control_id"] for item in nearest_form_by_block]),
        "nearest_structural_by_block": nearest_structural_by_block,
        "nearest_form_by_block": nearest_form_by_block,
        "first_structural_win_by_control": first_structural_win_by_control,
        "first_form_win_by_control": first_form_win_by_control,
        "best_structural_by_control": best_structural_by_control,
        "best_form_by_control": best_form_by_control,
    }

    if output_prefix is None:
        output_prefix = run_id

    output_root = output_dir or VALIDATION_DIR
    output_root.mkdir(parents=True, exist_ok=True)
    csv_path = output_root / f"{output_prefix}_vs_control_bank.csv"
    json_path = output_root / f"{output_prefix}_vs_control_bank.json"
    summary_path = output_root / f"{output_prefix}_vs_control_bank_chronology.json"

    fieldnames = list(rows[0].keys()) if rows else []
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "comparison_date": date.today().isoformat(),
                "run_manifest": str(run_manifest_path),
                "run_id": run_id,
                "rows": rows,
            },
            fh,
            ensure_ascii=False,
            indent=2,
        )

    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "comparison_date": date.today().isoformat(),
                "run_manifest": str(run_manifest_path),
                "run_id": run_id,
                "control_count": len(controls),
                "block_count": len(blocks),
                **summary,
            },
            fh,
            ensure_ascii=False,
            indent=2,
        )

    return {
        "csv_path": str(csv_path),
        "json_path": str(json_path),
        "summary_path": str(summary_path),
        "control_count": len(controls),
        "block_count": len(blocks),
        "summary": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare a completed run manifest against the null/control bank")
    parser.add_argument("--run-manifest", type=Path, required=True, help="Path to a completed long-run manifest.json")
    parser.add_argument("--output-prefix", type=str, default=None, help="Optional output filename prefix")
    parser.add_argument(
        "--control-id",
        dest="control_ids",
        action="append",
        default=None,
        help="Specific control id to include. Repeat to add more.",
    )
    parser.add_argument(
        "--block-id",
        dest="block_ids",
        action="append",
        default=None,
        help="Specific block id to include. Repeat to add more.",
    )
    args = parser.parse_args()

    result = compare_run_manifest_to_control_bank(
        run_manifest_path=args.run_manifest,
        output_prefix=args.output_prefix,
        control_ids=args.control_ids,
        block_ids=args.block_ids,
    )
    print(result["csv_path"])
    print(result["json_path"])
    print(result["summary_path"])


if __name__ == "__main__":
    main()
