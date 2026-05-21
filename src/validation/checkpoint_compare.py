"""
Checkpoint-to-Validator Comparison
==================================
Purpose:
    Compare selected bridge checkpoints from a retrodiction run against an
    attested historical validator corpus.

    This is the first historical-validation layer for the reinforced bridge
    workflow. It does not claim that a bridge is "correct." It reports where
    attested material lands relative to the generated path.

Usage:
    python -m src.validation.checkpoint_compare ^
        --run-summary data/retrodiction/french/v2_convergence/run_summary.json ^
        --validator data/processed/historical/old_french_tokens.json
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from src.retrodiction.engine_reinforced_v2 import (
    CHAR_BIGRAM_TOP_N,
    CHAR_TRIGRAM_TOP_N,
    SUFFIX_LEN,
    SUFFIX_TOP_N,
    _build_sparse_profile,
    _extract_char_ngrams_from_sequences,
    _extract_suffixes_from_sequences,
    _sparse_profile_cosine,
)
from src.retrodiction.similarity import (
    ReferenceSet,
    cosine_similarity,
    scaled_euclidean_distance,
    structural_vector,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VALIDATION_DIR = PROJECT_ROOT / "data" / "validation"


def _load_corpus_and_profiles(path: Path) -> tuple[dict, list[list[str]], dict[str, float], dict[str, float]]:
    with path.open(encoding="utf-8") as fh:
        corpus = json.load(fh)

    sequences = corpus["sequences"]
    stage_or_lang = path.stem.replace("_tokens", "")
    candidate_paths = [
        path.parent.parent / "matrices" / f"{stage_or_lang}_ngram_meta.json",
        path.parent.parent.parent / "matrices" / f"{stage_or_lang}_ngram_meta.json",
        PROJECT_ROOT / "data" / "matrices" / f"{stage_or_lang}_ngram_meta.json",
    ]
    ngram_meta_path = next((candidate for candidate in candidate_paths if candidate.exists()), None)
    if ngram_meta_path is None:
        raise FileNotFoundError(f"No ngram metadata found for {path}")
    with ngram_meta_path.open(encoding="utf-8") as fh:
        meta = json.load(fh)

    return corpus, sequences, meta["bigrams"], meta["trigrams"]


def default_checkpoint_stage_ids(run_summary: dict) -> list[str]:
    """
    Select a compact ladder spanning the run.

    For long runs this produces:
        start, quarter, midpoint, three-quarter, late-tail, endpoint
    """
    stages = run_summary.get("stages", [])
    if not stages:
        return []

    final_index = len(stages) - 1
    indices = {
        0,
        final_index // 4,
        final_index // 2,
        (3 * final_index) // 4,
        max(0, final_index - 3),
        final_index,
    }
    ordered = sorted(indices)
    return [stages[i]["stage_id"] for i in ordered]


@dataclass
class CorpusFormReference:
    label: str
    char_bigram_profile: dict[str, float]
    char_trigram_profile: dict[str, float]
    suffix_profile: dict[str, float]

    @classmethod
    def from_sequences(cls, label: str, sequences: list[list[str]]) -> "CorpusFormReference":
        return cls(
            label=label,
            char_bigram_profile=_build_sparse_profile(
                _extract_char_ngrams_from_sequences(sequences, 2),
                CHAR_BIGRAM_TOP_N,
            ),
            char_trigram_profile=_build_sparse_profile(
                _extract_char_ngrams_from_sequences(sequences, 3),
                CHAR_TRIGRAM_TOP_N,
            ),
            suffix_profile=_build_sparse_profile(
                _extract_suffixes_from_sequences(sequences, SUFFIX_LEN),
                SUFFIX_TOP_N,
            ),
        )

    def score(self, sequences: list[list[str]]) -> dict[str, float]:
        bg = _build_sparse_profile(_extract_char_ngrams_from_sequences(sequences, 2), CHAR_BIGRAM_TOP_N)
        tg = _build_sparse_profile(_extract_char_ngrams_from_sequences(sequences, 3), CHAR_TRIGRAM_TOP_N)
        sfx = _build_sparse_profile(_extract_suffixes_from_sequences(sequences, SUFFIX_LEN), SUFFIX_TOP_N)

        char_bigram_cos = _sparse_profile_cosine(bg, self.char_bigram_profile)
        char_trigram_cos = _sparse_profile_cosine(tg, self.char_trigram_profile)
        suffix_cos = _sparse_profile_cosine(sfx, self.suffix_profile)
        total = 0.40 * char_bigram_cos + 0.40 * char_trigram_cos + 0.20 * suffix_cos
        return {
            "validator_form_score": float(total),
            "validator_char_bigram_cosine": float(char_bigram_cos),
            "validator_char_trigram_cosine": float(char_trigram_cos),
            "validator_suffix_cosine": float(suffix_cos),
        }


def compare_run_to_validator(
    run_summary_path: Path,
    validator_tokens_path: Path,
    checkpoint_stage_ids: list[str] | None = None,
    output_path: Path | None = None,
) -> dict:
    with run_summary_path.open(encoding="utf-8") as fh:
        run_summary = json.load(fh)

    if checkpoint_stage_ids is None:
        checkpoint_stage_ids = default_checkpoint_stage_ids(run_summary)

    stages = run_summary.get("stages", [])
    stage_by_id = {stage["stage_id"]: stage for stage in stages}
    missing = [stage_id for stage_id in checkpoint_stage_ids if stage_id not in stage_by_id]
    if missing:
        raise ValueError(f"Checkpoint stage ids missing from run summary: {missing}")

    validator_meta, validator_sequences, validator_bigrams, validator_trigrams = _load_corpus_and_profiles(
        validator_tokens_path
    )
    validator_vec = structural_vector(validator_sequences, validator_bigrams, validator_trigrams)
    validator_form_ref = CorpusFormReference.from_sequences(
        validator_meta.get("language", validator_tokens_path.stem.replace("_tokens", "")),
        validator_sequences,
    )

    references = ReferenceSet()
    scale = references.real_language_scale

    comparisons = []
    for stage_id in checkpoint_stage_ids:
        stage = stage_by_id[stage_id]
        corpus_path = Path(stage["artifacts"]["corpus_json"])
        _, sequences, bigrams, trigrams = _load_corpus_and_profiles(corpus_path)
        vec = structural_vector(sequences, bigrams, trigrams)

        item = {
            "stage_id": stage["stage_id"],
            "iteration": stage["iteration"],
            "mutation_operator": stage["mutation_operator"],
            "preview_txt": stage["artifacts"].get("preview_txt"),
            "corpus_json": stage["artifacts"].get("corpus_json"),
            "latin_structural_score": stage.get("latin_structural_score"),
            "latin_form_score": stage.get("latin_form_score"),
            "total_score": stage.get("total_score"),
            "coherence_label": stage.get("diagnostics", {}).get("coherence_label"),
            "validator_structural_cosine": cosine_similarity(vec, validator_vec),
            "validator_structural_distance": scaled_euclidean_distance(vec, validator_vec, scale),
            "validator_structural_vector": [round(float(x), 6) for x in vec.tolist()],
        }
        item.update(validator_form_ref.score(sequences))
        comparisons.append(item)

    best_by_distance = min(comparisons, key=lambda item: item["validator_structural_distance"])
    best_by_form = max(comparisons, key=lambda item: item["validator_form_score"])
    best_by_cosine = max(comparisons, key=lambda item: item["validator_structural_cosine"])

    result = {
        "comparison_date": date.today().isoformat(),
        "run_summary": str(run_summary_path),
        "run_language": run_summary.get("language"),
        "run_algorithm": run_summary.get("algorithm"),
        "validator_tokens": str(validator_tokens_path),
        "validator_language": validator_meta.get("language"),
        "validator_branch_language": validator_meta.get("branch_language"),
        "validator_period": validator_meta.get("historical_period"),
        "checkpoint_stage_ids": checkpoint_stage_ids,
        "validator_structural_vector": [round(float(x), 6) for x in validator_vec.tolist()],
        "best_by_structural_distance": best_by_distance,
        "best_by_structural_cosine": best_by_cosine,
        "best_by_form_score": best_by_form,
        "comparisons": comparisons,
    }

    if output_path is None:
        VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
        run_label = f"{run_summary.get('language', 'run')}_{run_summary_path.parent.name}"
        validator_label = validator_meta.get("language", validator_tokens_path.stem.replace("_tokens", ""))
        output_path = VALIDATION_DIR / f"{run_label}_vs_{validator_label}.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    log.info("Wrote checkpoint comparison to %s", output_path)

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare bridge checkpoints against an attested validator")
    parser.add_argument("--run-summary", type=Path, required=True, help="Path to a run_summary.json")
    parser.add_argument("--validator", type=Path, required=True, help="Path to a historical *_tokens.json")
    parser.add_argument(
        "--stage-id",
        dest="stage_ids",
        action="append",
        default=None,
        help="Specific checkpoint stage id to include. Repeat to add more.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path")
    args = parser.parse_args()

    compare_run_to_validator(
        run_summary_path=args.run_summary,
        validator_tokens_path=args.validator,
        checkpoint_stage_ids=args.stage_ids,
        output_path=args.output,
    )
