"""
Post hoc dimensional robustness check for the v5 paper run.

Computes 15 corpus-level features for a fixed set of corpora and reports
4D / 10D / 15D cosine distances to Latin (z-scored), per-feature
diagnostics for Modern Portuguese, and PCA on the 15-feature matrix.

Pure post hoc analysis; does NOT touch production scoring.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.fingerprint.ngram import build_profile, extract_ngrams
from src.ingest.tokenize import corpus_stats
from src.retrodiction.engine_reinforced_v2 import (
    SUFFIX_LEN,
    SUFFIX_TOP_N,
    _build_sparse_profile,
    _extract_suffixes_from_sequences,
)
from src.retrodiction.similarity import (
    ReferenceSet,
    cosine_similarity,
    scaled_euclidean_distance,
    structural_vector,
    top_k_coverage,
)
from src.sequester.guard import (
    lock_sequestration,
    load_sequestered,
    unlock_sequestration,
)

PAPER_RUN_DIR = (
    PROJECT_ROOT
    / "data"
    / "retrodiction"
    / "french"
    / "v5_fortran_c16_seed45_paper_run"
)
HISTORICAL_DIR = PROJECT_ROOT / "data" / "processed" / "historical"
ROMANCE_DIR = PROJECT_ROOT / "data" / "processed" / "romance"
NULLS_DIR = PROJECT_ROOT / "data" / "processed" / "nulls"
SEQUESTERED_DIR = PROJECT_ROOT / "data" / "sequestered"
MATRICES_DIR = PROJECT_ROOT / "data" / "matrices"

OUT_DIR = PROJECT_ROOT / "data" / "validation" / "2026-05-19_dimensional_robustness"


def block_best_corpus(block_dir_name: str) -> Path:
    """
    Locate the best_corpus_json for a given block, by reading manifest.json.
    """
    manifest_path = PAPER_RUN_DIR / "manifest.json"
    with manifest_path.open(encoding="utf-8") as fh:
        manifest = json.load(fh)
    for block in manifest["blocks"]:
        if block["block"] == block_dir_name:
            return Path(block["best_corpus_json"])
    raise KeyError(block_dir_name)


def load_sequences_only(path: Path) -> list[list[str]]:
    with path.open(encoding="utf-8") as fh:
        corpus = json.load(fh)
    return corpus["sequences"]


def load_sequestered_sequences(name: str, reason: str) -> list[list[str]]:
    unlock_sequestration(reason)
    try:
        corpus = load_sequestered(name)
    finally:
        lock_sequestration()
    return corpus["sequences"]


def build_unigram_profile(sequences: list[list[str]], top_n: int) -> dict[str, float]:
    counter: Counter = Counter()
    for seq in sequences:
        counter.update(seq)
    top = counter.most_common(top_n)
    total = sum(c for _, c in top)
    if total == 0:
        return {}
    return {tok: c / total for tok, c in top}


def unigram_total_coverage(sequences: list[list[str]], top_n: int) -> float:
    """
    Fraction of ALL token mass captured by the top-N unigram types.
    Different from top_k_coverage on a top-5000 profile because the
    denominator here is total tokens (the full corpus), not top-N tokens.
    """
    counter: Counter = Counter()
    for seq in sequences:
        counter.update(seq)
    total = sum(counter.values())
    if total == 0:
        return 0.0
    top = counter.most_common(top_n)
    return sum(c for _, c in top) / total


def sequence_length_stats(sequences: list[list[str]]) -> tuple[float, float]:
    """Return (mean, variance) of sequence lengths in tokens."""
    if not sequences:
        return 0.0, 0.0
    lengths = np.array([len(s) for s in sequences], dtype=np.float64)
    return float(lengths.mean()), float(lengths.var(ddof=0))


def word_length_stats(sequences: list[list[str]]) -> tuple[float, float]:
    """Return (mean, variance) of word length in characters."""
    lengths = []
    for seq in sequences:
        for tok in seq:
            lengths.append(len(tok))
    if not lengths:
        return 0.0, 0.0
    arr = np.array(lengths, dtype=np.float64)
    return float(arr.mean()), float(arr.var(ddof=0))


def hapax_ratio(sequences: list[list[str]]) -> float:
    counter: Counter = Counter()
    for seq in sequences:
        counter.update(seq)
    if not counter:
        return 0.0
    hapax = sum(1 for v in counter.values() if v == 1)
    return hapax / len(counter)


def zipf_slope(sequences: list[list[str]], rmin: int = 10, rmax: int = 1000) -> float:
    """
    Least-squares slope of log(freq) vs log(rank) for ranks in [rmin, rmax].
    A clean Zipf is around -1. Heavier-tailed (analytic) tends toward < -1
    (steeper), flatter distributions toward > -1.
    """
    counter: Counter = Counter()
    for seq in sequences:
        counter.update(seq)
    if not counter:
        return 0.0
    freqs = sorted(counter.values(), reverse=True)
    upper = min(rmax, len(freqs))
    if upper < rmin + 5:
        return 0.0
    ranks = np.arange(rmin, upper + 1, dtype=np.float64)
    f = np.array(freqs[rmin - 1 : upper], dtype=np.float64)
    f = np.where(f <= 0, 1.0, f)
    x = np.log(ranks)
    y = np.log(f)
    slope, _ = np.polyfit(x, y, 1)
    return float(slope)


def profile_entropy_nats(profile: dict[str, float]) -> float:
    """
    Shannon entropy (nats) over a relative-frequency profile that has been
    renormalized to its top-N entries. Equivalent to the engine's own
    profile_entropy but accepting a sparse dict.
    """
    if not profile:
        return 0.0
    total = sum(profile.values())
    if total <= 0.0:
        return 0.0
    h = 0.0
    for v in profile.values():
        if v > 0.0:
            p = v / total
            h -= p * math.log(p)
    return h


def compute_features(sequences: list[list[str]]) -> dict[str, float]:
    """
    Compute the 15-feature vector for one corpus.

    Features 1-4 follow the EXACT 4D structural_vector() definition.
    Features 5-15 are new.
    """
    # --- Tier-0: exact reuse of existing 4D code ---
    # Build full top-5000 profiles for bigram/trigram, matching the existing
    # validator/control-bank load path (see control_bank_compare and
    # validator_bank_compare fallback in _load_sequences_and_profiles).
    bigram_counter = extract_ngrams(sequences, 2)
    trigram_counter = extract_ngrams(sequences, 3)
    bigram_profile_5000 = build_profile(bigram_counter, 5000)
    trigram_profile_5000 = build_profile(trigram_counter, 5000)

    vec4 = structural_vector(sequences, bigram_profile_5000, trigram_profile_5000)
    ttr = float(vec4[0])
    top100_bg = float(vec4[1])
    top100_tg = float(vec4[2])
    log_mean_seq_len = float(vec4[3])

    # --- Tier-1: higher-resolution variants on existing axes ---
    top500_bg = float(top_k_coverage(bigram_profile_5000, 500))
    top500_tg = float(top_k_coverage(trigram_profile_5000, 500))
    # Top-1000 unigram coverage: fraction of total token mass in top-1000 types
    top1000_uni = unigram_total_coverage(sequences, 1000)
    _, seq_len_var = sequence_length_stats(sequences)
    word_len_mean, word_len_var = word_length_stats(sequences)

    # --- Tier-2: orthogonal axes ---
    hapax = hapax_ratio(sequences)
    zipf = zipf_slope(sequences, 10, 1000)
    # Bigram/trigram entropy on the top-5000 profile (consistent with
    # similarity.py's existing entropy reference, but reported per-token
    # rather than normalized — z-scoring handles scale across corpora).
    bg_entropy = profile_entropy_nats(bigram_profile_5000)
    tg_entropy = profile_entropy_nats(trigram_profile_5000)
    # Suffix profile entropy uses the engine's own suffix infrastructure.
    suffix_profile = _build_sparse_profile(
        _extract_suffixes_from_sequences(sequences, SUFFIX_LEN),
        SUFFIX_TOP_N,
    )
    sfx_entropy = profile_entropy_nats(suffix_profile)

    return {
        "f01_ttr": ttr,
        "f02_top100_bigram_coverage": top100_bg,
        "f03_top100_trigram_coverage": top100_tg,
        "f04_log_mean_seq_len": log_mean_seq_len,
        "f05_top500_bigram_coverage": top500_bg,
        "f06_top500_trigram_coverage": top500_tg,
        "f07_top1000_unigram_coverage": top1000_uni,
        "f08_seq_length_variance": seq_len_var,
        "f09_word_length_mean": word_len_mean,
        "f10_word_length_variance": word_len_var,
        "f11_hapax_ratio": hapax,
        "f12_zipf_slope_10_1000": zipf,
        "f13_bigram_entropy": bg_entropy,
        "f14_trigram_entropy": tg_entropy,
        "f15_suffix_entropy": sfx_entropy,
    }


# ---------------------------------------------------------------------------
# Corpus catalogue
# ---------------------------------------------------------------------------

SEQ_REASON = (
    "Post hoc dimensional robustness check (2026-05-19): rescoring v5 paper "
    "run against existing corpora using a richer feature space. No production "
    "scoring path is modified."
)


def load_all_corpora() -> dict[str, list[list[str]]]:
    """
    Return a dict {label: sequences} for every corpus we want to score.
    """
    corpora: dict[str, list[list[str]]] = {}

    # Target
    corpora["latin"] = load_sequestered_sequences("latin", SEQ_REASON)

    # Validators
    for name in ("old_french", "middle_french", "anglo_norman",
                 "langue_d_oil", "old_spanish", "old_occitan"):
        corpora[name] = load_sequences_only(HISTORICAL_DIR / f"{name}_tokens.json")

    # Controls
    corpora["markov"] = load_sequences_only(NULLS_DIR / "markov" / "markov_tokens.json")
    corpora["sumerian"] = load_sequences_only(NULLS_DIR / "sumerian" / "sumerian_tokens.json")
    corpora["portuguese_withheld"] = load_sequestered_sequences("portuguese", SEQ_REASON)

    # Source + key blocks
    corpora["modern_french_source"] = load_sequences_only(ROMANCE_DIR / "french_tokens.json")
    for block in ("block_0001", "block_0338", "block_0424", "block_0474",
                  "block_0613", "block_0614"):
        corpora[f"v5_{block}"] = load_sequences_only(block_best_corpus(block))

    return corpora


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def sanity_check_4d_distance(
    block_0474_seqs: list[list[str]],
    portuguese_seqs: list[list[str]],
) -> dict:
    """
    Reproduce the existing scaled_euclidean_distance between Modern Portuguese
    and block 0474 (expected ~0.9730).
    """
    refs = ReferenceSet()
    scale = refs.real_language_scale

    # Block vector
    bg_b = build_profile(extract_ngrams(block_0474_seqs, 2), 5000)
    tg_b = build_profile(extract_ngrams(block_0474_seqs, 3), 5000)
    vec_b = structural_vector(block_0474_seqs, bg_b, tg_b)

    # Portuguese vector
    bg_p = build_profile(extract_ngrams(portuguese_seqs, 2), 5000)
    tg_p = build_profile(extract_ngrams(portuguese_seqs, 3), 5000)
    vec_p = structural_vector(portuguese_seqs, bg_p, tg_p)

    sed = scaled_euclidean_distance(vec_b, vec_p, scale)
    cos = cosine_similarity(vec_b, vec_p)
    return {
        "block_0474_4d_vec": vec_b.tolist(),
        "portuguese_4d_vec": vec_p.tolist(),
        "scale_real_language": scale.tolist(),
        "scaled_euclidean_distance": sed,
        "cosine_similarity": cos,
    }


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    cs = cosine_similarity(a, b)
    return 1.0 - cs


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(">> Loading corpora", flush=True)
    corpora = load_all_corpora()

    # ---- Step 1: sanity check ----
    print(">> Step 1: sanity check (Portuguese vs block 0474)", flush=True)
    sanity = sanity_check_4d_distance(
        corpora["v5_block_0474"], corpora["portuguese_withheld"]
    )
    print(json.dumps(sanity, indent=2), flush=True)

    # ---- Step 2: compute features ----
    print(">> Step 2: computing 15-feature vectors for all corpora", flush=True)
    feature_table: dict[str, dict[str, float]] = {}
    for label, seqs in corpora.items():
        print(f"   - {label} (n_seqs={len(seqs)})", flush=True)
        feature_table[label] = compute_features(seqs)

    feature_names = list(next(iter(feature_table.values())).keys())
    # Z-score across corpora
    matrix = np.array(
        [[feature_table[label][f] for f in feature_names] for label in feature_table],
        dtype=np.float64,
    )
    labels = list(feature_table.keys())
    means = matrix.mean(axis=0)
    stds = matrix.std(axis=0, ddof=0)
    stds = np.where(stds < 1e-12, 1.0, stds)
    z = (matrix - means) / stds

    # Persist raw features
    feat_csv = OUT_DIR / "features.csv"
    with feat_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["corpus"] + feature_names + [f"{f}_z" for f in feature_names])
        for i, label in enumerate(labels):
            writer.writerow(
                [label]
                + [feature_table[label][f] for f in feature_names]
                + [float(z[i, j]) for j in range(len(feature_names))]
            )
    print(f">> Wrote {feat_csv}", flush=True)

    # ---- Step 3+4: distances to Latin under 4D, 10D, 15D ----
    print(">> Step 3+4: 4D / 10D / 15D cosine distances to Latin", flush=True)
    latin_idx = labels.index("latin")
    dims = {"4D": slice(0, 4), "10D": slice(0, 10), "15D": slice(0, 15)}
    distances: dict[str, dict[str, float]] = {label: {} for label in labels}
    for name, sl in dims.items():
        latin_vec = z[latin_idx, sl]
        for i, label in enumerate(labels):
            distances[label][name] = cosine_distance(z[i, sl], latin_vec)

    # Build the comparison table (sorted by 15D distance ascending = nearest first)
    rank_table = []
    sorted_labels = sorted(labels, key=lambda lbl: distances[lbl]["15D"])
    # Compute ranks within each dimensionality.
    # Latin itself is always rank 0 / distance 0 — keep it in the table to make
    # that visible but exclude it from rank shuffles by ranking from 0.
    rank_in: dict[str, dict[str, int]] = {label: {} for label in labels}
    for name in dims:
        order = sorted(labels, key=lambda lbl: distances[lbl][name])
        for r, lbl in enumerate(order):
            rank_in[lbl][name] = r

    dist_csv = OUT_DIR / "distances.csv"
    with dist_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "corpus", "dist_4D", "dist_10D", "dist_15D",
            "rank_4D", "rank_10D", "rank_15D",
            "rank_shift_4D_to_15D",
        ])
        for label in sorted_labels:
            r4 = rank_in[label]["4D"]
            r10 = rank_in[label]["10D"]
            r15 = rank_in[label]["15D"]
            rank_table.append({
                "corpus": label,
                "dist_4D": distances[label]["4D"],
                "dist_10D": distances[label]["10D"],
                "dist_15D": distances[label]["15D"],
                "rank_4D": r4,
                "rank_10D": r10,
                "rank_15D": r15,
                "rank_shift_4D_to_15D": r15 - r4,
            })
            writer.writerow([
                label,
                distances[label]["4D"],
                distances[label]["10D"],
                distances[label]["15D"],
                r4, r10, r15,
                r15 - r4,
            ])
    print(f">> Wrote {dist_csv}", flush=True)
    print(json.dumps(rank_table, indent=2), flush=True)

    # ---- Step 5: Portuguese per-feature diagnostic ----
    print(">> Step 5: Portuguese per-feature contributions", flush=True)
    pt_idx = labels.index("portuguese_withheld")
    # New features are 5..15 (i.e. indices 4..14)
    pt_z = z[pt_idx]
    lat_z = z[latin_idx]
    pt_vec = z[pt_idx, 0:15]
    lat_vec = z[latin_idx, 0:15]
    # Cosine similarity = (pt · lat) / (|pt|*|lat|).
    # Per-feature contribution to the dot product (before normalization)
    # is pt_z[i] * lat_z[i].
    pt_norm = float(np.linalg.norm(pt_vec))
    lat_norm = float(np.linalg.norm(lat_vec))
    contribs = []
    for j, fname in enumerate(feature_names):
        prod = float(pt_z[j] * lat_z[j])
        contribs.append({
            "feature": fname,
            "z_portuguese": float(pt_z[j]),
            "z_latin": float(lat_z[j]),
            "dot_product_term": prod,
            "normalized_term": prod / (pt_norm * lat_norm) if pt_norm and lat_norm else 0.0,
        })
    # Restrict reporting to features 5..15 (the new ones), per the spec.
    new_feature_contribs = contribs[4:]
    diag_path = OUT_DIR / "portuguese_per_feature.json"
    with diag_path.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "pt_norm": pt_norm,
                "lat_norm": lat_norm,
                "cosine_similarity_15D": float(
                    np.dot(pt_vec, lat_vec) / (pt_norm * lat_norm)
                ) if pt_norm and lat_norm else 0.0,
                "all_features": contribs,
                "new_features_only": new_feature_contribs,
            },
            fh,
            indent=2,
        )
    print(f">> Wrote {diag_path}", flush=True)
    for c in new_feature_contribs:
        print(
            f"   {c['feature']:35s}  z_PT={c['z_portuguese']:+7.3f}  "
            f"z_LAT={c['z_latin']:+7.3f}  prod={c['dot_product_term']:+8.4f}",
            flush=True,
        )

    # ---- Step 6: PCA on z-scored 15-feature matrix ----
    print(">> Step 6: PCA on 15-feature matrix", flush=True)
    # z has mean ~0 / std 1 per feature already. Use SVD directly.
    u, s, vt = np.linalg.svd(z, full_matrices=False)
    # Variance explained = s^2 / (n - 1)
    eig = (s ** 2) / max(1, z.shape[0] - 1)
    total = eig.sum()
    explained = eig / total
    cumulative = np.cumsum(explained)
    pca_summary = {
        "n_corpora": int(z.shape[0]),
        "n_features": int(z.shape[1]),
        "explained_variance_ratio": [float(x) for x in explained],
        "cumulative_explained": [float(x) for x in cumulative],
        "n_pcs_for_95pct": int(np.searchsorted(cumulative, 0.95) + 1),
        "n_pcs_for_99pct": int(np.searchsorted(cumulative, 0.99) + 1),
    }
    pca_path = OUT_DIR / "pca.json"
    with pca_path.open("w", encoding="utf-8") as fh:
        json.dump(pca_summary, fh, indent=2)
    print(f">> Wrote {pca_path}", flush=True)
    print(json.dumps(pca_summary, indent=2), flush=True)

    # Persist sanity check too
    sanity_path = OUT_DIR / "sanity_check.json"
    with sanity_path.open("w", encoding="utf-8") as fh:
        json.dump(sanity, fh, indent=2)
    print(f">> Wrote {sanity_path}", flush=True)


if __name__ == "__main__":
    main()
