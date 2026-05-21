"""
Validator Bank Comparison
=========================

Compare one selected checkpoint per block from a completed retrodiction run
against every active attested validator corpus, then summarize where the run
is nearest in structure and form over time.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from src.fingerprint.ngram import build_profile, extract_ngrams
from src.validation.checkpoint_compare import (
    CorpusFormReference,
    _load_corpus_and_profiles,
)
from src.retrodiction.similarity import (
    ReferenceSet,
    cosine_similarity,
    scaled_euclidean_distance,
    structural_vector,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VALIDATION_DIR = PROJECT_ROOT / "data" / "validation"
VALIDATOR_BANK_MANIFEST = PROJECT_ROOT / "data" / "raw" / "historical" / "validator_bank_manifest.csv"


@dataclass(frozen=True)
class ValidatorBankRow:
    corpus_id: str
    branch_language: str
    historical_period: str
    region: str
    status: str
    attested_only: str
    input_dir: str
    notes: str
    date_start: int
    date_end: int
    order_hint: int


def _load_validator_bank_rows(path: Path = VALIDATOR_BANK_MANIFEST) -> list[ValidatorBankRow]:
    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        return [
            ValidatorBankRow(
                corpus_id=row["corpus_id"].strip(),
                branch_language=row["branch_language"].strip(),
                historical_period=row["historical_period"].strip(),
                region=row["region"].strip(),
                status=row["status"].strip(),
                attested_only=row["attested_only"].strip(),
                input_dir=row["input_dir"].strip(),
                notes=row.get("notes", "").strip(),
                date_start=int(row["date_start"]),
                date_end=int(row["date_end"]),
                order_hint=int(row["order_hint"]),
            )
            for row in reader
        ]


def _validator_tokens_path(corpus_id: str) -> Path:
    return PROJECT_ROOT / "data" / "processed" / "historical" / f"{corpus_id}_tokens.json"


def _load_sequences_and_profiles(path: Path) -> tuple[list[list[str]], dict[str, float], dict[str, float]]:
    with path.open(encoding="utf-8") as fh:
        corpus = json.load(fh)

    sequences = corpus["sequences"]
    try:
        _, _, bigrams, trigrams = _load_corpus_and_profiles(path)
        return sequences, bigrams, trigrams
    except FileNotFoundError:
        return sequences, build_profile(extract_ngrams(sequences, 2), 5000), build_profile(extract_ngrams(sequences, 3), 5000)


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


def compare_run_manifest_to_validator_bank(
    run_manifest_path: Path,
    output_prefix: str | None = None,
    validator_ids: list[str] | None = None,
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
    validators = [
        row for row in _load_validator_bank_rows()
        if row.status == "active" and row.attested_only.lower() == "yes" and _validator_tokens_path(row.corpus_id).exists()
    ]
    if validator_ids is not None:
        selected = set(validator_ids)
        validators = [row for row in validators if row.corpus_id in selected]

    references = ReferenceSet()
    scale = references.real_language_scale

    validator_refs: dict[str, dict] = {}
    for validator in validators:
        validator_path = _validator_tokens_path(validator.corpus_id)
        validator_meta, validator_sequences, validator_bigrams, validator_trigrams = _load_corpus_and_profiles(validator_path)
        validator_refs[validator.corpus_id] = {
            "bank": validator,
            "meta": validator_meta,
            "vec": structural_vector(validator_sequences, validator_bigrams, validator_trigrams),
            "form_ref": CorpusFormReference.from_sequences(validator.corpus_id, validator_sequences),
            "tokens_path": str(validator_path),
        }

    rows: list[dict] = []
    for block in blocks:
        stage_id = block["best_stage_id"]
        corpus_path = Path(block["best_corpus_json"])
        sequences, bigrams, trigrams = _load_sequences_and_profiles(corpus_path)
        vec = structural_vector(sequences, bigrams, trigrams)

        for validator in validators:
            validator_ref = validator_refs[validator.corpus_id]
            form_scores = validator_ref["form_ref"].score(sequences)
            rows.append(
                {
                    "run_id": run_id,
                    "block_id": block["block"],
                    "stage_id": stage_id,
                    "mutation_operator": None,
                    "validator_corpus": validator.corpus_id,
                    "validator_language": validator_ref["meta"].get("language", validator.corpus_id),
                    "validator_branch_language": validator.branch_language,
                    "validator_period": validator.historical_period,
                    "validator_region": validator.region,
                    "validator_date_start": validator.date_start,
                    "validator_date_end": validator.date_end,
                    "validator_order_hint": validator.order_hint,
                    "latin_structural_score": block.get("final_latin_structural_score"),
                    "latin_form_score": block.get("final_latin_form_score"),
                    "family_alignment_score": block.get("final_family_alignment_score"),
                    "coherence_label": block.get("final_coherence_label"),
                    "validator_structural_cosine": cosine_similarity(vec, validator_ref["vec"]),
                    "validator_structural_distance": scaled_euclidean_distance(vec, validator_ref["vec"], scale),
                    "validator_form_score": form_scores["validator_form_score"],
                    "validator_char_bigram_cosine": form_scores["validator_char_bigram_cosine"],
                    "validator_char_trigram_cosine": form_scores["validator_char_trigram_cosine"],
                    "validator_suffix_cosine": form_scores["validator_suffix_cosine"],
                    "corpus_json": str(corpus_path),
                    "preview_txt": block.get("best_preview_txt"),
                }
            )

    rows.sort(key=lambda item: (_block_sort_key(item["block_id"]), item["validator_corpus"]))

    by_block: dict[str, list[dict]] = {}
    for row in rows:
        by_block.setdefault(row["block_id"], []).append(row)

    nearest_structural_by_block = []
    nearest_form_by_block = []
    for block_id in sorted(by_block, key=_block_sort_key):
        block_rows = by_block[block_id]
        nearest_structural_by_block.append(min(block_rows, key=lambda item: item["validator_structural_distance"]))
        nearest_form_by_block.append(max(block_rows, key=lambda item: item["validator_form_score"]))

    first_structural_win_by_validator: dict[str, dict] = {}
    first_form_win_by_validator: dict[str, dict] = {}
    for item in nearest_structural_by_block:
        first_structural_win_by_validator.setdefault(item["validator_corpus"], item)
    for item in nearest_form_by_block:
        first_form_win_by_validator.setdefault(item["validator_corpus"], item)

    best_structural_by_validator = {}
    best_form_by_validator = {}
    for validator in validators:
        validator_rows = [row for row in rows if row["validator_corpus"] == validator.corpus_id]
        best_structural_by_validator[validator.corpus_id] = min(
            validator_rows, key=lambda item: item["validator_structural_distance"]
        )
        best_form_by_validator[validator.corpus_id] = max(
            validator_rows, key=lambda item: item["validator_form_score"]
        )

    chronology_summary = {
        "structural_path": _dedupe_path([item["validator_corpus"] for item in nearest_structural_by_block]),
        "form_path": _dedupe_path([item["validator_corpus"] for item in nearest_form_by_block]),
        "nearest_structural_by_block": nearest_structural_by_block,
        "nearest_form_by_block": nearest_form_by_block,
        "first_structural_win_by_validator": first_structural_win_by_validator,
        "first_form_win_by_validator": first_form_win_by_validator,
        "best_structural_by_validator": best_structural_by_validator,
        "best_form_by_validator": best_form_by_validator,
    }

    if output_prefix is None:
        output_prefix = run_id

    output_root = output_dir or VALIDATION_DIR
    output_root.mkdir(parents=True, exist_ok=True)
    csv_path = output_root / f"{output_prefix}_vs_validator_bank.csv"
    json_path = output_root / f"{output_prefix}_vs_validator_bank.json"
    summary_path = output_root / f"{output_prefix}_vs_validator_bank_chronology.json"

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
                "validator_count": len(validators),
                "block_count": len(blocks),
                **chronology_summary,
            },
            fh,
            ensure_ascii=False,
            indent=2,
        )

    return {
        "csv_path": str(csv_path),
        "json_path": str(json_path),
        "summary_path": str(summary_path),
        "validator_count": len(validators),
        "block_count": len(blocks),
        "summary": chronology_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare a completed run manifest against the attested validator bank")
    parser.add_argument("--run-manifest", type=Path, required=True, help="Path to a completed long-run manifest.json")
    parser.add_argument("--output-prefix", type=str, default=None, help="Optional output filename prefix")
    args = parser.parse_args()

    result = compare_run_manifest_to_validator_bank(
        run_manifest_path=args.run_manifest,
        output_prefix=args.output_prefix,
    )
    print(result["csv_path"])
    print(result["json_path"])
    print(result["summary_path"])


if __name__ == "__main__":
    main()
