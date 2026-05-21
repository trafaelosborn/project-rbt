"""
Self-similarity calibration for Project RBT scoring functions.

For each corpus (Latin, validators, controls), compute the three scoring
functions with the corpus as both reference and candidate. Reports the
achievable ceiling for each metric under the production code path.

Does NOT modify production scoring code. All paths call into the existing
production functions (LatinReference, LatinFormReference, hungarian_alignment).
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Production scoring imports — these are the exact classes the engine instantiates.
from src.fingerprint.ngram import build_profile, extract_ngrams
from src.retrodiction.engine_reinforced import LatinReference, REWARD_SCORE_SCALE
from src.retrodiction.engine_reinforced_v2 import (
    LatinFormReference,
    SUFFIX_LEN,
    SUFFIX_TOP_N,
    CHAR_BIGRAM_TOP_N,
    CHAR_TRIGRAM_TOP_N,
    _build_sparse_profile,
    _extract_char_ngrams_from_sequences,
    _extract_suffixes_from_sequences,
    _sparse_profile_cosine,
)
from src.retrodiction.similarity import (
    cosine_similarity,
    scaled_euclidean_distance,
    structural_vector,
    ReferenceSet,
)
from src.sequester.guard import (
    load_sequestered,
    lock_sequestration,
    unlock_sequestration,
)
from src.validation.checkpoint_compare import (
    CorpusFormReference,
    _load_corpus_and_profiles,
)
from src.validation.hungarian_alignment import (
    FamilyAlignmentConfig,
    extract_family_inventory,
    hungarian_alignment_diagnostics,
    load_latin_family_reference,
)

OUT_DIR = PROJECT_ROOT / "data" / "validation" / "2026-05-19_self_similarity"

PAPER_RUN_DIR = (
    PROJECT_ROOT / "data" / "retrodiction" / "french" / "v5_fortran_c16_seed45_paper_run"
)
BLOCK_0614_CORPUS = (
    PAPER_RUN_DIR / "blocks" / "block_0614" / "corpora" / "FR_v5_001_tokens.json"
)

HISTORICAL_DIR = PROJECT_ROOT / "data" / "processed" / "historical"
NULLS_DIR = PROJECT_ROOT / "data" / "processed" / "nulls"
SEQUESTERED_REASON = (
    "Self-similarity calibration (2026-05-19): reading sequestered Latin/Portuguese "
    "corpus tokens for post hoc scoring ceiling. No mutation, no search."
)


# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------

def _load_tokens(path: Path) -> list[list[str]]:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)["sequences"]


def _load_sequestered(name: str) -> list[list[str]]:
    unlock_sequestration(SEQUESTERED_REASON)
    try:
        corpus = load_sequestered(name)
    finally:
        lock_sequestration()
    return corpus["sequences"]


def load_corpus_sequences() -> dict[str, list[list[str]]]:
    return {
        "latin": _load_sequestered("latin"),
        "old_french": _load_tokens(HISTORICAL_DIR / "old_french_tokens.json"),
        "middle_french": _load_tokens(HISTORICAL_DIR / "middle_french_tokens.json"),
        "anglo_norman": _load_tokens(HISTORICAL_DIR / "anglo_norman_tokens.json"),
        "langue_d_oil": _load_tokens(HISTORICAL_DIR / "langue_d_oil_tokens.json"),
        "old_spanish": _load_tokens(HISTORICAL_DIR / "old_spanish_tokens.json"),
        "old_occitan": _load_tokens(HISTORICAL_DIR / "old_occitan_tokens.json"),
        "markov": _load_tokens(NULLS_DIR / "markov" / "markov_tokens.json"),
        "sumerian": _load_tokens(NULLS_DIR / "sumerian" / "sumerian_tokens.json"),
        "portuguese_withheld": _load_sequestered("portuguese"),
    }


# ---------------------------------------------------------------------------
# Step 1: smoke test against validator-bank artifact
# ---------------------------------------------------------------------------

EXPECTED_OLD_FRENCH_BLOCK_0614 = {
    "validator_structural_cosine": 0.9992140862166782,
    "validator_structural_distance": 2.024154993404628,
    "validator_form_score": 0.38375984210715847,
    "validator_char_bigram_cosine": 0.5788081845561647,
    "validator_char_trigram_cosine": 0.30216729257656105,
    "validator_suffix_cosine": 0.15684825627034088,
}


def reproduce_old_french_block_0614() -> dict:
    """
    Recompute the validator-bank row for (block_0614, old_french) using the
    same code path validator_bank_compare uses, and compare against the
    persisted CSV value.
    """
    block_corpus_path = BLOCK_0614_CORPUS
    validator_path = HISTORICAL_DIR / "old_french_tokens.json"

    _, block_seqs, block_bg, block_tg = _load_corpus_and_profiles(block_corpus_path)
    _, val_seqs, val_bg, val_tg = _load_corpus_and_profiles(validator_path)

    block_vec = structural_vector(block_seqs, block_bg, block_tg)
    val_vec = structural_vector(val_seqs, val_bg, val_tg)

    refs = ReferenceSet()
    scale = refs.real_language_scale

    struct_cos = cosine_similarity(block_vec, val_vec)
    struct_sed = scaled_euclidean_distance(block_vec, val_vec, scale)

    form_ref = CorpusFormReference.from_sequences("old_french", val_seqs)
    form_scores = form_ref.score(block_seqs)

    computed = {
        "validator_structural_cosine": struct_cos,
        "validator_structural_distance": struct_sed,
        "validator_form_score": form_scores["validator_form_score"],
        "validator_char_bigram_cosine": form_scores["validator_char_bigram_cosine"],
        "validator_char_trigram_cosine": form_scores["validator_char_trigram_cosine"],
        "validator_suffix_cosine": form_scores["validator_suffix_cosine"],
    }

    deltas = {k: abs(computed[k] - EXPECTED_OLD_FRENCH_BLOCK_0614[k]) for k in computed}
    return {
        "expected": EXPECTED_OLD_FRENCH_BLOCK_0614,
        "computed": computed,
        "max_abs_delta": max(deltas.values()),
        "deltas": deltas,
        "passes": max(deltas.values()) < 1e-9,
    }


# ---------------------------------------------------------------------------
# Step 2: Latin self-similarity via PRODUCTION scoring functions
# ---------------------------------------------------------------------------

def latin_self_via_production() -> dict:
    """
    Score Latin against itself using the same classes the engine instantiates
    in src/retrodiction/engine_reinforced_v2.py:
        - LatinReference (structural)
        - LatinFormReference (form)
        - load_latin_family_reference + extract_family_inventory + hungarian_alignment_diagnostics

    For each one we report what would happen if a "Latin" corpus were the
    candidate, under two reasonable definitions of "Latin candidate":
        (a) the same Latin slice the reference was built from (the [:50_000]
            slice that LatinReference/LatinFormReference/family ref all use)
        (b) the full Latin corpus

    (a) is the theoretical ceiling — identical inputs on both sides. (b) is
    what an engine candidate exactly equal to the full Latin source would
    score, exposing any asymmetry built into the reference path.
    """
    # Production references — exactly what the engine builds.
    latin_struct_ref = LatinReference()
    latin_form_ref = LatinFormReference()
    cfg = FamilyAlignmentConfig()
    latin_family_ref = load_latin_family_reference(cfg)

    # Reference slice = sequences[:50_000] for all three references (all three
    # internally truncate to 50k). Load it once.
    unlock_sequestration(SEQUESTERED_REASON)
    try:
        latin_corpus = load_sequestered("latin")
    finally:
        lock_sequestration()
    full_latin_seqs = latin_corpus["sequences"]
    matched_slice = full_latin_seqs[: 50_000]

    # ------ Structural (production: LatinReference.score) ------
    # Build a candidate structural_vector exactly the way the engine does it
    # in _evaluate_sequences: build top-5000 bg/tg profiles from candidate
    # sequences and pass to structural_vector().
    def _candidate_struct_vec(seqs: list[list[str]]) -> np.ndarray:
        bg = build_profile(extract_ngrams(seqs, 2), 5000)
        tg = build_profile(extract_ngrams(seqs, 3), 5000)
        return structural_vector(seqs, bg, tg)

    cand_vec_matched = _candidate_struct_vec(matched_slice)
    cand_vec_full = _candidate_struct_vec(full_latin_seqs)
    ref_vec = latin_struct_ref.vec
    # Production score: -REWARD_SCORE_SCALE * ||cand[:3] - latin_reward[:3]||
    struct_matched_production = latin_struct_ref.score(cand_vec_matched)
    struct_full_production = latin_struct_ref.score(cand_vec_full)
    # Alternate definition cited in the paper and the calibration brief.
    struct_matched_cos_minus_1 = cosine_similarity(cand_vec_matched, ref_vec) - 1.0
    struct_full_cos_minus_1 = cosine_similarity(cand_vec_full, ref_vec) - 1.0

    # ------ Form (production: LatinFormReference.score) ------
    form_matched = latin_form_ref.score(matched_slice)
    form_full = latin_form_ref.score(full_latin_seqs)

    # ------ Family alignment (production: hungarian_alignment_diagnostics) ------
    cand_family_matched = extract_family_inventory("latin", matched_slice, cfg)
    cand_family_full = extract_family_inventory("latin", full_latin_seqs, cfg)
    fam_matched = hungarian_alignment_diagnostics(cand_family_matched, latin_family_ref, cfg)
    fam_full = hungarian_alignment_diagnostics(cand_family_full, latin_family_ref, cfg)

    return {
        "latin_total_sequences": len(full_latin_seqs),
        "reference_slice_size": len(matched_slice),
        "structural": {
            "production_formula": "-REWARD_SCORE_SCALE * ||cand[:3] - latin_reward[:3]||",
            "paper_formula": "cosine_similarity(cand_vec, latin_vec) - 1",
            "candidate_eq_matched_slice": {
                "production_score": struct_matched_production,
                "cosine_minus_1": struct_matched_cos_minus_1,
                "candidate_vec": cand_vec_matched.tolist(),
                "reference_vec": ref_vec.tolist(),
            },
            "candidate_eq_full_latin": {
                "production_score": struct_full_production,
                "cosine_minus_1": struct_full_cos_minus_1,
                "candidate_vec": cand_vec_full.tolist(),
                "reference_vec": ref_vec.tolist(),
            },
            "reward_score_scale": float(REWARD_SCORE_SCALE),
        },
        "form": {
            "candidate_eq_matched_slice": form_matched,
            "candidate_eq_full_latin": form_full,
            "weights": {
                "char_bigram": latin_form_ref.char_bigram_weight,
                "char_trigram": latin_form_ref.char_trigram_weight,
                "suffix": latin_form_ref.suffix_weight,
            },
        },
        "family": {
            "candidate_eq_matched_slice": {
                "family_alignment_score": fam_matched["family_alignment_score"],
                "family_alignment_cost": fam_matched["family_alignment_cost"],
                "matched_family_count": fam_matched["matched_family_count"],
                "unmatched_bridge_families": fam_matched["unmatched_bridge_families"],
                "unmatched_reference_families": fam_matched["unmatched_reference_families"],
                "bridge_family_count": fam_matched["bridge_family_count"],
                "reference_family_count": fam_matched["reference_family_count"],
            },
            "candidate_eq_full_latin": {
                "family_alignment_score": fam_full["family_alignment_score"],
                "family_alignment_cost": fam_full["family_alignment_cost"],
                "matched_family_count": fam_full["matched_family_count"],
                "unmatched_bridge_families": fam_full["unmatched_bridge_families"],
                "unmatched_reference_families": fam_full["unmatched_reference_families"],
                "bridge_family_count": fam_full["bridge_family_count"],
                "reference_family_count": fam_full["reference_family_count"],
            },
        },
    }


# ---------------------------------------------------------------------------
# Step 3: per-corpus self-similarity
# ---------------------------------------------------------------------------

def corpus_self_similarity(label: str, sequences: list[list[str]]) -> dict:
    """
    For an arbitrary corpus, build per-corpus structural / form / family
    references using the same code paths the production engine uses for
    Latin, then score the corpus against itself.

    Three definitions of "self":
        (a) structural: cosine_similarity(vec, vec) - 1 and the engine's
            production -scale*||vec[:3]-vec[:3]|| analogue (both should be 0)
        (b) form: CorpusFormReference (the validator-bank form scorer that
            uses the same profile machinery as LatinFormReference, just
            without the [:50_000] truncation)
        (c) family: extract_family_inventory + hungarian_alignment_diagnostics
            with the inventory on both sides
    """
    # ------ Structural ------
    bg = build_profile(extract_ngrams(sequences, 2), 5000)
    tg = build_profile(extract_ngrams(sequences, 3), 5000)
    vec = structural_vector(sequences, bg, tg)
    struct_cos_minus_1 = cosine_similarity(vec, vec) - 1.0
    struct_production_analogue = -REWARD_SCORE_SCALE * float(
        np.linalg.norm(vec[:3] - vec[:3])
    )

    # ------ Form (CorpusFormReference is the production validator-bank scorer) ------
    form_ref = CorpusFormReference.from_sequences(label, sequences)
    form_scores = form_ref.score(sequences)

    # ------ Family ------
    cfg = FamilyAlignmentConfig()
    inv = extract_family_inventory(label, sequences, cfg)
    fam = hungarian_alignment_diagnostics(inv, inv, cfg)

    return {
        "label": label,
        "n_sequences": len(sequences),
        "n_tokens": int(sum(len(s) for s in sequences)),
        "structural_cos_minus_1": struct_cos_minus_1,
        "structural_production_analogue": struct_production_analogue,
        "form_score": form_scores["validator_form_score"],
        "form_char_bigram_cosine": form_scores["validator_char_bigram_cosine"],
        "form_char_trigram_cosine": form_scores["validator_char_trigram_cosine"],
        "form_suffix_cosine": form_scores["validator_suffix_cosine"],
        "family_alignment_score": fam["family_alignment_score"],
        "family_matched_count": fam["matched_family_count"],
        "family_unmatched_bridge": fam["unmatched_bridge_families"],
        "family_unmatched_ref": fam["unmatched_reference_families"],
        "family_bridge_count": fam["bridge_family_count"],
        "family_ref_count": fam["reference_family_count"],
        "structural_vec": vec.tolist(),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(">> Step 1: smoke test (block_0614 vs old_french)", flush=True)
    smoke = reproduce_old_french_block_0614()
    smoke_path = OUT_DIR / "smoke_test.json"
    with smoke_path.open("w", encoding="utf-8") as fh:
        json.dump(smoke, fh, indent=2)
    print(f"   max_abs_delta = {smoke['max_abs_delta']:.3e} (passes={smoke['passes']})", flush=True)

    print(">> Step 2: Latin self-similarity via production scoring path", flush=True)
    latin_self = latin_self_via_production()
    latin_self_path = OUT_DIR / "latin_self_production.json"
    with latin_self_path.open("w", encoding="utf-8") as fh:
        json.dump(latin_self, fh, indent=2)
    print(json.dumps(
        {
            "structural_production_matched": latin_self["structural"]["candidate_eq_matched_slice"]["production_score"],
            "structural_cos_minus_1_matched": latin_self["structural"]["candidate_eq_matched_slice"]["cosine_minus_1"],
            "structural_production_full": latin_self["structural"]["candidate_eq_full_latin"]["production_score"],
            "structural_cos_minus_1_full": latin_self["structural"]["candidate_eq_full_latin"]["cosine_minus_1"],
            "form_matched": latin_self["form"]["candidate_eq_matched_slice"]["latin_form_score"],
            "form_full": latin_self["form"]["candidate_eq_full_latin"]["latin_form_score"],
            "family_matched": latin_self["family"]["candidate_eq_matched_slice"]["family_alignment_score"],
            "family_full": latin_self["family"]["candidate_eq_full_latin"]["family_alignment_score"],
        },
        indent=2,
    ), flush=True)

    print(">> Step 3: per-corpus self-similarity", flush=True)
    corpora = load_corpus_sequences()
    rows: list[dict] = []
    for label, seqs in corpora.items():
        print(f"   - {label} (n_seqs={len(seqs)})", flush=True)
        rows.append(corpus_self_similarity(label, seqs))

    csv_path = OUT_DIR / "calibration.csv"
    fields = [
        "label", "n_sequences", "n_tokens",
        "structural_cos_minus_1", "structural_production_analogue",
        "form_score", "form_char_bigram_cosine", "form_char_trigram_cosine", "form_suffix_cosine",
        "family_alignment_score", "family_matched_count",
        "family_unmatched_bridge", "family_unmatched_ref",
        "family_bridge_count", "family_ref_count",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(fields)
        for r in rows:
            writer.writerow([r[f] for f in fields])
    print(f">> Wrote {csv_path}", flush=True)

    json_path = OUT_DIR / "calibration.json"
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump({"rows": rows, "latin_self_production": latin_self, "smoke_test": smoke}, fh, indent=2)
    print(f">> Wrote {json_path}", flush=True)

    # Print compact table.
    print()
    print("Per-corpus self-similarity:")
    print(f"  {'corpus':25s} {'struct(cos-1)':>14s} {'struct(prod)':>14s} {'form':>10s} {'family':>10s}")
    for r in rows:
        print(
            f"  {r['label']:25s} {r['structural_cos_minus_1']:>+14.6e} {r['structural_production_analogue']:>+14.6e} {r['form_score']:>10.6f} {r['family_alignment_score']:>10.6f}"
        )


if __name__ == "__main__":
    main()
