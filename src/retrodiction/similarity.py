"""
Structural Similarity
=====================
Purpose:
    Compute vocabulary-independent similarity scores between a generated
    intermediate corpus and the reference corpora (Markov noise, Sumerian).

    All features are computed from raw token sequences and frequency profiles,
    without reference to specific token identities. This means the same metric
    applies equally to French, Sumerian, Markov noise, and any generated
    intermediate — the vocabularies are irrelevant.

Structural feature vector (4 dimensions):
    [0] type_token_ratio         — morphological complexity proxy
    [1] top100_bigram_coverage   — fraction of bigram mass in top-100 bigrams
    [2] top100_trigram_coverage  — fraction of trigram mass in top-100 trigrams
    [3] log_mean_seq_len         — log(1 + mean sequence length)

    top-k coverage measures CONCENTRATION of the bigram distribution:
        HIGH  = analytic grammar (few patterns dominate: article+noun, prep+article)
        LOW   = synthetic grammar (any word can follow any other, near-uniform)
        ~0.02 = fully uniform (Markov noise with top-5000 profile: 100/5000)

    As retrodiction mixes the bigram model toward uniform, coverage FALLS.
    This is the primary discriminating signal between analytic and synthetic
    grammar, and between Romance intermediates and the Markov floor.

    Shannon entropy was rejected because it saturates near log(V) ≈ 8.52 for
    all V=5000 profiles, making cosine similarity unable to discriminate.
    See docs/decisions/011_similarity_metric.md.

Scoring convention:
    All scores are cosine similarities in structural feature space, range [0, 1].
    Interpretation:
        vs_markov_noise:  LOW = more structured than noise (good)
        vs_sumerian:      diagnostic — how similar to real non-IE structure

    A valid intermediate stage should have vs_markov_noise declining as
    retrodiction progresses (structure dissolving toward uniformity).

References:
    Reference vectors are loaded on demand from the processed null corpora and
    cached in the ReferenceSet instance.
"""

import json
import math
import logging
from functools import cached_property
from pathlib import Path

import numpy as np

from src.ingest.tokenize import corpus_stats

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MATRICES_DIR = PROJECT_ROOT / "data" / "matrices"

TOP_K = 100   # number of top n-grams for coverage feature
REAL_LANGUAGE_REFERENCE_LANGS = (
    "french",
    "italian",
    "spanish",
    "romanian",
    "occitan",
    "genoese",
    "sumerian",
)
COHERENCE_MARGIN_THRESHOLD = 1.0
EPSILON = 1e-9


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------

def top_k_coverage(profile: dict[str, float], k: int = TOP_K) -> float:
    """
    Fraction of total n-gram mass accounted for by the top-k most frequent
    n-grams in the profile.

    For a fully uniform profile of N entries: coverage = k / N.
    For a fully concentrated profile: coverage approaches 1.0.

    Args:
        profile: Relative frequency dict (values sum to 1.0).
        k:       Number of top entries to sum.

    Returns:
        float in [0, 1].
    """
    if not profile:
        return 0.0
    top_k_vals = sorted(profile.values(), reverse=True)[:k]
    return float(sum(top_k_vals))


def profile_entropy(profile: dict[str, float]) -> float:
    """
    Shannon entropy of a relative-frequency profile.

    Args:
        profile: Relative frequency dict (values sum to 1.0).

    Returns:
        Entropy in nats.
    """
    return float(sum(-p * math.log(p) for p in profile.values() if p > 0.0))


def structural_vector(
    sequences: list[list[str]],
    bigram_profile: dict[str, float],
    trigram_profile: dict[str, float],
    k: int = TOP_K,
) -> np.ndarray:
    """
    Compute a 4-dimensional vocabulary-independent structural feature vector.

    Args:
        sequences:       Token sequences (used for TTR and mean length).
        bigram_profile:  Relative bigram frequency dict (sums to 1.0).
        trigram_profile: Relative trigram frequency dict (sums to 1.0).
        k:               Number of top n-grams for coverage features.

    Returns:
        np.ndarray shape (4,):
            [ttr, top_k_bigram_coverage, top_k_trigram_coverage, log_mean_seq_len]
    """
    stats = corpus_stats(sequences)
    return np.array([
        stats["type_token_ratio"],
        top_k_coverage(bigram_profile, k),
        top_k_coverage(trigram_profile, k),
        math.log1p(stats["mean_seq_length"]),
    ], dtype=np.float64)


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Cosine similarity between two vectors. Returns 0.0 if either is zero."""
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0.0 or n2 == 0.0:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))


def scaled_euclidean_distance(
    v1: np.ndarray,
    v2: np.ndarray,
    scale: np.ndarray,
) -> float:
    """
    Euclidean distance after per-feature scaling.

    This is useful when structural features live on different numeric scales,
    such as TTR vs log mean sequence length.
    """
    v1 = np.asarray(v1, dtype=np.float64)
    v2 = np.asarray(v2, dtype=np.float64)
    scale = np.maximum(np.asarray(scale, dtype=np.float64), EPSILON)
    return float(np.linalg.norm((v1 - v2) / scale))


def coherence_diagnostics(
    vec: np.ndarray,
    real_language_centroid: np.ndarray,
    markov_vec: np.ndarray,
    feature_scale: np.ndarray,
    coherent_margin_threshold: float = COHERENCE_MARGIN_THRESHOLD,
) -> dict[str, float | str]:
    """
    Diagnose whether a stage remains language-like or is drifting toward noise.

    The stage is evaluated in scaled structural-vector space against:
        1. the centroid of attested real-language references, and
        2. the Markov noise floor.

    Positive margin means the stage is closer to the real-language manifold than
    to Markov noise. Large positive margin indicates a comfortably coherent
    intermediate rather than optimizer junk.
    """
    dist_real = scaled_euclidean_distance(vec, real_language_centroid, feature_scale)
    dist_markov = scaled_euclidean_distance(vec, markov_vec, feature_scale)
    margin = dist_markov - dist_real

    if margin >= coherent_margin_threshold:
        label = "coherent"
    elif margin >= 0.0:
        label = "borderline"
    else:
        label = "noise_like"

    return {
        "distance_to_real_language_centroid": dist_real,
        "distance_to_markov_noise": dist_markov,
        "language_likeness_margin": margin,
        "coherence_label": label,
    }


# ---------------------------------------------------------------------------
# Reference set
# ---------------------------------------------------------------------------

class ReferenceSet:
    """
    Precomputed structural vectors for Markov noise and Sumerian.
    Loaded lazily from the processed corpus files on first access.
    """

    def __init__(self) -> None:
        pass

    def _load_corpus(self, path: Path) -> tuple[list[list[str]], dict, dict]:
        """Load sequences and ngram meta from a corpus JSON and its ngram meta."""
        with path.open(encoding="utf-8") as fh:
            corpus = json.load(fh)
        sequences = corpus["sequences"]

        lang = path.stem.replace("_tokens", "")
        ngram_meta_path = path.parent.parent.parent / "matrices" / f"{lang}_ngram_meta.json"
        if not ngram_meta_path.exists():
            ngram_meta_path = MATRICES_DIR / f"{lang}_ngram_meta.json"
        with ngram_meta_path.open(encoding="utf-8") as fh:
            meta = json.load(fh)

        return sequences, meta["bigrams"], meta["trigrams"]

    @cached_property
    def markov(self) -> np.ndarray:
        path = PROCESSED_DIR / "nulls" / "markov" / "markov_tokens.json"
        sequences, bigrams, trigrams = self._load_corpus(path)
        vec = structural_vector(sequences, bigrams, trigrams)
        log.info("Markov reference vector: %s", vec)
        return vec

    @cached_property
    def sumerian(self) -> np.ndarray:
        path = PROCESSED_DIR / "nulls" / "sumerian" / "sumerian_tokens.json"
        sequences, bigrams, trigrams = self._load_corpus(path)
        vec = structural_vector(sequences, bigrams, trigrams)
        log.info("Sumerian reference vector: %s", vec)
        return vec

    @cached_property
    def real_language_vectors(self) -> np.ndarray:
        """
        Structural vectors for attested non-noise languages available in-repo.
        """
        vectors = []
        for lang in REAL_LANGUAGE_REFERENCE_LANGS:
            if lang == "sumerian":
                path = PROCESSED_DIR / "nulls" / "sumerian" / "sumerian_tokens.json"
            else:
                path = PROCESSED_DIR / "romance" / f"{lang}_tokens.json"
            sequences, bigrams, trigrams = self._load_corpus(path)
            vectors.append(structural_vector(sequences, bigrams, trigrams))
        arr = np.vstack(vectors)
        log.info("Loaded %d real-language reference vectors", arr.shape[0])
        return arr

    @cached_property
    def real_language_centroid(self) -> np.ndarray:
        return self.real_language_vectors.mean(axis=0)

    @cached_property
    def real_language_scale(self) -> np.ndarray:
        scale = self.real_language_vectors.std(axis=0, ddof=0)
        return np.maximum(scale, EPSILON)

    def score(
        self,
        sequences: list[list[str]],
        bigram_profile: dict[str, float],
        trigram_profile: dict[str, float],
    ) -> dict[str, float | None]:
        """
        Score a generated corpus against the reference corpora.

        Returns a dict matching the bridge stage record scores format.
        vs_portuguese_control and vs_latin_ground_truth are always None here.
        """
        vec = structural_vector(sequences, bigram_profile, trigram_profile)
        return {
            "vs_markov_noise": cosine_similarity(vec, self.markov),
            "vs_sumerian": cosine_similarity(vec, self.sumerian),
            "vs_portuguese_control": None,
            "vs_latin_ground_truth": None,
        }

    def coherence_from_vector(self, vec: np.ndarray) -> dict[str, float | str]:
        """
        Diagnose whether a structural vector remains closer to attested language
        space than to the Markov noise floor.
        """
        return coherence_diagnostics(
            vec,
            self.real_language_centroid,
            self.markov,
            self.real_language_scale,
        )
