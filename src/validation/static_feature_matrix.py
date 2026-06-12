"""
V2 static feature matrix calibration.

This module computes the preregistered v2 corpus-level feature matrix from
fixed corpora only. It is validation/preregistration tooling, not a
retrodiction search path.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from src.fingerprint.ngram import build_profile, extract_ngrams
from src.ingest.tokenize import corpus_stats
from src.retrodiction.engine_reinforced_v2 import (
    SUFFIX_LEN,
    SUFFIX_TOP_N,
    _build_sparse_profile,
    _extract_suffixes_from_sequences,
)
from src.retrodiction.similarity import profile_entropy, top_k_coverage
from src.sequester.guard import load_sequestered, lock_sequestration, unlock_sequestration

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "data" / "v2" / "corpora_manifest.json"
DEFAULT_MATRIX_PATH = PROJECT_ROOT / "data" / "v2" / "static_feature_matrix.csv"
DEFAULT_SPEARMAN_PATH = PROJECT_ROOT / "data" / "v2" / "static_feature_spearman.csv"
DEFAULT_TIERS_PATH = PROJECT_ROOT / "data" / "v2" / "static_feature_tiers.json"

DEFAULT_CORPUS_BUDGET = 20_000
DEFAULT_SAMPLE_SEED = 1729
COLLINEARITY_THRESHOLD = 0.85

SEQUESTRATION_REASON = (
    "H1 preregistration static feature calibration: fixed corpus fingerprints "
    "for feature tier selection before any v2 run trajectories are inspected."
)

VOWELS = frozenset(
    "aeiouy"
    "àáâãäåæèéêëìíîïòóôõöøùúûüýÿœ"
    "аеёиоуыэюяіїєў"
    "آاويىة"
)


@dataclass(frozen=True)
class FeatureSpec:
    feature_id: str
    column: str
    label: str
    prereg_tier: str


REGISTERED_FEATURES: tuple[FeatureSpec, ...] = (
    FeatureSpec("F01", "f01_ttr", "Type-token ratio", "diagnostic"),
    FeatureSpec("F02", "f02_top100_bigram_coverage", "Top-100 bigram coverage", "in_loop_candidate"),
    FeatureSpec("F03", "f03_top100_trigram_coverage", "Top-100 trigram coverage", "diagnostic"),
    FeatureSpec("F04", "f04_log_mean_seq_len", "Log mean sequence length", "excluded_unless_length_ops"),
    FeatureSpec("F05", "f05_top500_bigram_coverage", "Top-500 bigram coverage", "diagnostic"),
    FeatureSpec("F06", "f06_top500_trigram_coverage", "Top-500 trigram coverage", "diagnostic"),
    FeatureSpec("F07", "f07_top1000_unigram_coverage", "Top-1000 unigram coverage", "diagnostic"),
    FeatureSpec("F08", "f08_seq_length_variance", "Sequence-length variance", "diagnostic"),
    FeatureSpec("F09", "f09_word_length_mean", "Mean word length", "in_loop_candidate"),
    FeatureSpec("F10", "f10_word_length_variance", "Word-length variance", "diagnostic"),
    FeatureSpec("F11", "f11_hapax_ratio", "Hapax ratio", "diagnostic"),
    FeatureSpec("F12", "f12_zipf_slope_10_1000", "Zipf slope ranks 10-1000", "diagnostic"),
    FeatureSpec("F13", "f13_bigram_entropy", "Bigram entropy", "diagnostic_or_f02_alternative"),
    FeatureSpec("F14", "f14_trigram_entropy", "Trigram entropy", "diagnostic"),
    FeatureSpec("F15", "f15_suffix_entropy", "Suffix entropy", "in_loop_candidate"),
    FeatureSpec("F16", "f16_heaps_beta", "Heaps' law exponent", "in_loop_candidate"),
    FeatureSpec("F17", "f17_conditional_char_entropy", "Conditional character entropy", "in_loop_candidate"),
    FeatureSpec("F18", "f18_dislegomena_to_hapax", "Dis-legomena to hapax ratio", "in_loop_candidate"),
    FeatureSpec("F19", "f19_word_edge_entropy_asymmetry", "Word-edge character asymmetry", "in_loop_candidate"),
    FeatureSpec("F20", "f20_consonant_vowel_ratio", "Consonant-vowel ratio", "diagnostic"),
    FeatureSpec("F21", "f21_sentence_length_skewness", "Sentence-length skewness", "diagnostic"),
)

AUXILIARY_FEATURES: tuple[FeatureSpec, ...] = (
    FeatureSpec(
        "AUX01",
        "aux_sentence_length_excess_kurtosis",
        "Sentence-length excess kurtosis",
        "auxiliary_diagnostic",
    ),
)

PROPOSED_IN_LOOP_COLUMNS = (
    "f02_top100_bigram_coverage",
    "f09_word_length_mean",
    "f15_suffix_entropy",
    "f16_heaps_beta",
    "f17_conditional_char_entropy",
    "f18_dislegomena_to_hapax",
    "f19_word_edge_entropy_asymmetry",
)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _load_manifest(manifest_path: Path = DEFAULT_MANIFEST_PATH) -> dict[str, Any]:
    return _read_json(manifest_path)


def _entry_is_static_matrix_row(entry: dict[str, Any], include_optional: bool = True) -> bool:
    if entry.get("status") != "ready":
        return False
    if entry.get("required_for_h1", False):
        return True
    return include_optional and entry.get("role") == "typological_control_optional"


def select_manifest_entries(
    manifest: dict[str, Any],
    include_optional: bool = True,
) -> list[dict[str, Any]]:
    return [
        entry
        for entry in manifest.get("corpora", [])
        if _entry_is_static_matrix_row(entry, include_optional=include_optional)
    ]


def _load_sequences_for_entry(entry: dict[str, Any], project_root: Path = PROJECT_ROOT) -> list[list[str]]:
    corpus_id = entry["corpus_id"]
    if entry.get("sequestered"):
        unlock_sequestration(SEQUESTRATION_REASON)
        try:
            corpus = load_sequestered(corpus_id)
        finally:
            lock_sequestration()
        return corpus["sequences"]

    tokens_path = project_root / entry["tokens_path"]
    corpus = _read_json(tokens_path)
    return corpus["sequences"]


def _stable_seed(base_seed: int, corpus_id: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{corpus_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False) % (2**32)


def subsample_sequences(
    sequences: list[list[str]],
    *,
    token_budget: int,
    seed: int,
    corpus_id: str,
) -> list[list[str]]:
    if token_budget <= 0:
        raise ValueError("token_budget must be positive")
    total_tokens = sum(len(seq) for seq in sequences)
    if total_tokens < token_budget:
        raise ValueError(
            f"Corpus '{corpus_id}' has {total_tokens} tokens, below budget {token_budget}"
        )

    rng = np.random.default_rng(_stable_seed(seed, corpus_id))
    indices = rng.permutation(len(sequences))
    selected: list[list[str]] = []
    seen = 0
    for raw_idx in indices:
        seq = list(sequences[int(raw_idx)])
        if not seq:
            continue
        remaining = token_budget - seen
        if len(seq) <= remaining:
            selected.append(seq)
            seen += len(seq)
        else:
            selected.append(seq[:remaining])
            seen += remaining
        if seen >= token_budget:
            break
    if seen != token_budget:
        raise ValueError(f"Could not sample {token_budget} tokens from '{corpus_id}'")
    return selected


def _token_counter(sequences: list[list[str]]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for seq in sequences:
        counter.update(seq)
    return counter


def _unigram_total_coverage(sequences: list[list[str]], top_n: int) -> float:
    counter = _token_counter(sequences)
    total = sum(counter.values())
    if total == 0:
        return 0.0
    return float(sum(count for _, count in counter.most_common(top_n)) / total)


def _variance(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(values.var(ddof=0))


def _sequence_length_moments(sequences: list[list[str]]) -> dict[str, float]:
    if not sequences:
        return {
            "mean": 0.0,
            "variance": 0.0,
            "skewness": 0.0,
            "excess_kurtosis": 0.0,
        }
    lengths = np.array([len(seq) for seq in sequences], dtype=np.float64)
    mean = float(lengths.mean())
    centered = lengths - mean
    variance = float(np.mean(centered**2))
    if variance <= 0.0:
        return {
            "mean": mean,
            "variance": 0.0,
            "skewness": 0.0,
            "excess_kurtosis": 0.0,
        }
    std = math.sqrt(variance)
    skewness = float(np.mean((centered / std) ** 3))
    excess_kurtosis = float(np.mean((centered / std) ** 4) - 3.0)
    return {
        "mean": mean,
        "variance": variance,
        "skewness": skewness,
        "excess_kurtosis": excess_kurtosis,
    }


def _word_length_stats(sequences: list[list[str]]) -> tuple[float, float]:
    lengths = [len(tok) for seq in sequences for tok in seq]
    if not lengths:
        return 0.0, 0.0
    arr = np.array(lengths, dtype=np.float64)
    return float(arr.mean()), _variance(arr)


def _hapax_ratio(counter: Counter[str]) -> float:
    if not counter:
        return 0.0
    return float(sum(1 for count in counter.values() if count == 1) / len(counter))


def _dislegomena_to_hapax(counter: Counter[str]) -> float:
    hapax = sum(1 for count in counter.values() if count == 1)
    dislegomena = sum(1 for count in counter.values() if count == 2)
    return float(dislegomena / max(hapax, 1))


def _zipf_slope(counter: Counter[str], rmin: int = 10, rmax: int = 1000) -> float:
    if not counter:
        return 0.0
    freqs = sorted(counter.values(), reverse=True)
    upper = min(rmax, len(freqs))
    if upper < rmin + 5:
        return 0.0
    ranks = np.arange(rmin, upper + 1, dtype=np.float64)
    y = np.log(np.array(freqs[rmin - 1 : upper], dtype=np.float64))
    x = np.log(ranks)
    slope, _ = np.polyfit(x, y, 1)
    return float(slope)


def _suffix_entropy(sequences: list[list[str]]) -> float:
    suffix_profile = _build_sparse_profile(
        _extract_suffixes_from_sequences(sequences, SUFFIX_LEN),
        SUFFIX_TOP_N,
    )
    return profile_entropy(suffix_profile)


def _heaps_beta(sequences: list[list[str]], sample_count: int = 12) -> float:
    tokens = [tok for seq in sequences for tok in seq]
    total = len(tokens)
    if total < 100:
        return 0.0
    sample_points = np.unique(
        np.linspace(max(50, total // (sample_count * 2)), total, sample_count).astype(int)
    )
    vocab: set[str] = set()
    next_point_idx = 0
    rows: list[tuple[int, int]] = []
    for idx, tok in enumerate(tokens, start=1):
        vocab.add(tok)
        while next_point_idx < len(sample_points) and idx >= sample_points[next_point_idx]:
            rows.append((int(sample_points[next_point_idx]), len(vocab)))
            next_point_idx += 1
    rows = [(n, v) for n, v in rows if n > 1 and v > 1]
    if len(rows) < 3:
        return 0.0
    x = np.log(np.array([n for n, _ in rows], dtype=np.float64))
    y = np.log(np.array([v for _, v in rows], dtype=np.float64))
    beta, _ = np.polyfit(x, y, 1)
    return float(beta)


def _conditional_char_entropy(sequences: list[list[str]]) -> float:
    transitions: dict[str, Counter[str]] = {}
    for seq in sequences:
        for tok in seq:
            for idx in range(len(tok) - 1):
                current = tok[idx]
                nxt = tok[idx + 1]
                transitions.setdefault(current, Counter())[nxt] += 1
    total = sum(sum(counter.values()) for counter in transitions.values())
    if total == 0:
        return 0.0
    entropy = 0.0
    for counter in transitions.values():
        subtotal = sum(counter.values())
        conditional = 0.0
        for count in counter.values():
            p = count / subtotal
            conditional -= p * math.log(p)
        entropy += (subtotal / total) * conditional
    return float(entropy)


def _entropy_from_counter(counter: Counter[str]) -> float:
    total = sum(counter.values())
    if total <= 0:
        return 0.0
    entropy = 0.0
    for count in counter.values():
        p = count / total
        entropy -= p * math.log(p)
    return float(entropy)


def _word_edge_entropy_asymmetry(sequences: list[list[str]]) -> float:
    initial: Counter[str] = Counter()
    final: Counter[str] = Counter()
    for seq in sequences:
        for tok in seq:
            if tok:
                initial[tok[0]] += 1
                final[tok[-1]] += 1
    return float(_entropy_from_counter(final) - _entropy_from_counter(initial))


def _consonant_vowel_ratio(sequences: list[list[str]]) -> float:
    vowels = 0
    consonants = 0
    for seq in sequences:
        for tok in seq:
            for ch in tok.lower():
                if not ch.isalpha():
                    continue
                if ch in VOWELS:
                    vowels += 1
                else:
                    consonants += 1
    return float(consonants / max(vowels, 1))


def compute_feature_vector(sequences: list[list[str]]) -> dict[str, float]:
    stats = corpus_stats(sequences)
    token_counts = _token_counter(sequences)
    bigram_profile = build_profile(extract_ngrams(sequences, 2), 5000)
    trigram_profile = build_profile(extract_ngrams(sequences, 3), 5000)
    seq_moments = _sequence_length_moments(sequences)
    word_mean, word_var = _word_length_stats(sequences)

    return {
        "f01_ttr": float(stats["type_token_ratio"]),
        "f02_top100_bigram_coverage": top_k_coverage(bigram_profile, 100),
        "f03_top100_trigram_coverage": top_k_coverage(trigram_profile, 100),
        "f04_log_mean_seq_len": math.log1p(float(stats["mean_seq_length"])),
        "f05_top500_bigram_coverage": top_k_coverage(bigram_profile, 500),
        "f06_top500_trigram_coverage": top_k_coverage(trigram_profile, 500),
        "f07_top1000_unigram_coverage": _unigram_total_coverage(sequences, 1000),
        "f08_seq_length_variance": seq_moments["variance"],
        "f09_word_length_mean": word_mean,
        "f10_word_length_variance": word_var,
        "f11_hapax_ratio": _hapax_ratio(token_counts),
        "f12_zipf_slope_10_1000": _zipf_slope(token_counts),
        "f13_bigram_entropy": profile_entropy(bigram_profile),
        "f14_trigram_entropy": profile_entropy(trigram_profile),
        "f15_suffix_entropy": _suffix_entropy(sequences),
        "f16_heaps_beta": _heaps_beta(sequences),
        "f17_conditional_char_entropy": _conditional_char_entropy(sequences),
        "f18_dislegomena_to_hapax": _dislegomena_to_hapax(token_counts),
        "f19_word_edge_entropy_asymmetry": _word_edge_entropy_asymmetry(sequences),
        "f20_consonant_vowel_ratio": _consonant_vowel_ratio(sequences),
        "f21_sentence_length_skewness": seq_moments["skewness"],
        "aux_sentence_length_excess_kurtosis": seq_moments["excess_kurtosis"],
    }


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        avg_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = avg_rank
        start = end
    return ranks


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def spearman_correlations(
    rows: list[dict[str, Any]],
    feature_columns: list[str],
) -> dict[tuple[str, str], float]:
    matrix = np.array([[float(row[col]) for col in feature_columns] for row in rows], dtype=np.float64)
    ranked = np.apply_along_axis(_rankdata, 0, matrix)
    correlations: dict[tuple[str, str], float] = {}
    for i, left in enumerate(feature_columns):
        for j, right in enumerate(feature_columns):
            correlations[(left, right)] = _pearson(ranked[:, i], ranked[:, j])
    return correlations


def _feature_by_column() -> dict[str, FeatureSpec]:
    return {feature.column: feature for feature in (*REGISTERED_FEATURES, *AUXILIARY_FEATURES)}


def _rho(correlations: dict[tuple[str, str], float], left: str, right: str) -> float:
    return correlations.get((left, right), correlations.get((right, left), 0.0))


def select_in_loop_tier(
    correlations: dict[tuple[str, str], float],
    threshold: float = COLLINEARITY_THRESHOLD,
) -> dict[str, Any]:
    retained: list[str] = []
    demoted: list[dict[str, Any]] = []
    for candidate in PROPOSED_IN_LOOP_COLUMNS:
        conflicts = [
            {
                "retained_feature": existing,
                "spearman_rho": _rho(correlations, candidate, existing),
            }
            for existing in retained
            if abs(_rho(correlations, candidate, existing)) >= threshold
        ]
        if conflicts:
            strongest = max(conflicts, key=lambda item: abs(float(item["spearman_rho"])))
            demoted.append(
                {
                    "feature": candidate,
                    "reason": "collinear_with_retained_in_loop_feature",
                    "strongest_conflict": strongest,
                }
            )
        else:
            retained.append(candidate)

    feature_lookup = _feature_by_column()
    registered_columns = [feature.column for feature in REGISTERED_FEATURES]
    diagnostic = [
        column
        for column in registered_columns
        if column not in retained
    ]
    return {
        "proposed_in_loop": list(PROPOSED_IN_LOOP_COLUMNS),
        "retained_in_loop": retained,
        "demoted_from_in_loop": demoted,
        "diagnostic_or_excluded": diagnostic,
        "retained_in_loop_labels": [
            {
                "feature_id": feature_lookup[column].feature_id,
                "column": column,
                "label": feature_lookup[column].label,
            }
            for column in retained
        ],
    }


def build_static_rows(
    *,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    project_root: Path = PROJECT_ROOT,
    token_budget: int = DEFAULT_CORPUS_BUDGET,
    sample_seed: int = DEFAULT_SAMPLE_SEED,
    include_optional: bool = True,
) -> list[dict[str, Any]]:
    manifest = _load_manifest(manifest_path)
    entries = select_manifest_entries(manifest, include_optional=include_optional)
    rows: list[dict[str, Any]] = []
    for entry in entries:
        sequences = _load_sequences_for_entry(entry, project_root=project_root)
        sampled = subsample_sequences(
            sequences,
            token_budget=token_budget,
            seed=sample_seed,
            corpus_id=entry["corpus_id"],
        )
        features = compute_feature_vector(sampled)
        rows.append(
            {
                "corpus_id": entry["corpus_id"],
                "language": entry["language"],
                "role": entry["role"],
                "family": entry["family"],
                "required_for_h1": entry.get("required_for_h1", False),
                "corpus_budget": token_budget,
                "sample_seed": sample_seed,
                "sampled_sequence_count": len(sampled),
                **features,
            }
        )
    return rows


def write_matrix_csv(rows: list[dict[str, Any]], path: Path) -> Path:
    feature_columns = [feature.column for feature in (*REGISTERED_FEATURES, *AUXILIARY_FEATURES)]
    fieldnames = [
        "corpus_id",
        "language",
        "role",
        "family",
        "required_for_h1",
        "corpus_budget",
        "sample_seed",
        "sampled_sequence_count",
        *feature_columns,
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_spearman_csv(
    correlations: dict[tuple[str, str], float],
    feature_columns: list[str],
    path: Path,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["feature", *feature_columns])
        for left in feature_columns:
            writer.writerow([left, *[correlations[(left, right)] for right in feature_columns]])
    return path


def write_tiers_json(
    *,
    rows: list[dict[str, Any]],
    correlations: dict[tuple[str, str], float],
    token_budget: int,
    sample_seed: int,
    matrix_path: Path,
    spearman_path: Path,
    output_path: Path,
) -> Path:
    tier_selection = select_in_loop_tier(correlations)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "project_rbt_v2_static_feature_tiers_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "corpus_budget": token_budget,
        "sample_seed": sample_seed,
        "corpus_count": len(rows),
        "matrix_path": str(matrix_path),
        "spearman_path": str(spearman_path),
        "collinearity_threshold_abs_spearman": COLLINEARITY_THRESHOLD,
        "f21_resolution": (
            "F21 is registered as sentence-length skewness. Sentence-length "
            "excess kurtosis is emitted only as auxiliary diagnostic AUX01."
        ),
        "in_loop_selection_policy": (
            "Evaluate proposed in-loop features in preregistered priority order; "
            "retain a candidate only if its absolute Spearman rho is below the "
            "threshold against every already-retained in-loop feature. The order "
            "favours direct, trainable typological axes when size-robust but more "
            "indirect diagnostics are collinear with them."
        ),
        "feature_specs": [feature.__dict__ for feature in REGISTERED_FEATURES],
        "auxiliary_diagnostics": [feature.__dict__ for feature in AUXILIARY_FEATURES],
        **tier_selection,
    }
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    return output_path


def build_and_write_static_artifacts(
    *,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    matrix_path: Path = DEFAULT_MATRIX_PATH,
    spearman_path: Path = DEFAULT_SPEARMAN_PATH,
    tiers_path: Path = DEFAULT_TIERS_PATH,
    token_budget: int = DEFAULT_CORPUS_BUDGET,
    sample_seed: int = DEFAULT_SAMPLE_SEED,
    include_optional: bool = True,
) -> dict[str, Any]:
    rows = build_static_rows(
        manifest_path=manifest_path,
        token_budget=token_budget,
        sample_seed=sample_seed,
        include_optional=include_optional,
    )
    feature_columns = [feature.column for feature in REGISTERED_FEATURES]
    correlations = spearman_correlations(rows, feature_columns)
    write_matrix_csv(rows, matrix_path)
    write_spearman_csv(correlations, feature_columns, spearman_path)
    write_tiers_json(
        rows=rows,
        correlations=correlations,
        token_budget=token_budget,
        sample_seed=sample_seed,
        matrix_path=matrix_path,
        spearman_path=spearman_path,
        output_path=tiers_path,
    )
    return {
        "matrix_path": str(matrix_path),
        "spearman_path": str(spearman_path),
        "tiers_path": str(tiers_path),
        "corpus_count": len(rows),
        "token_budget": token_budget,
        "sample_seed": sample_seed,
        "retained_in_loop": select_in_loop_tier(correlations)["retained_in_loop"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build v2 static feature calibration artifacts")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX_PATH)
    parser.add_argument("--spearman", type=Path, default=DEFAULT_SPEARMAN_PATH)
    parser.add_argument("--tiers", type=Path, default=DEFAULT_TIERS_PATH)
    parser.add_argument("--budget", type=int, default=DEFAULT_CORPUS_BUDGET)
    parser.add_argument("--seed", type=int, default=DEFAULT_SAMPLE_SEED)
    parser.add_argument("--exclude-optional", action="store_true")
    args = parser.parse_args(argv)

    result = build_and_write_static_artifacts(
        manifest_path=args.manifest,
        matrix_path=args.matrix,
        spearman_path=args.spearman,
        tiers_path=args.tiers,
        token_budget=args.budget,
        sample_seed=args.seed,
        include_optional=not args.exclude_optional,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
