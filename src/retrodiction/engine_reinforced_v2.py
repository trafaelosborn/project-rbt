"""
Relational Reinforced Retrodiction V2
=====================================
Purpose:
    A wider-search-space reinforced engine that mutates actual corpora rather
    than only perturbing a fixed transition matrix.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.fingerprint.ngram import build_profile, extract_ngrams
from src.retrodiction.engine_reinforced import (
    LANG_CODES,
    LatinReference,
    _fingerprint_sequences,
    _save_stage_corpus,
)
from src.retrodiction.similarity import ReferenceSet, structural_vector
from src.sequester.guard import lock_sequestration, unlock_sequestration

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RETRODICTION_DIR = PROJECT_ROOT / "data" / "retrodiction"

V2_UNLOCK_REASON = (
    "Phase 3 reinforcement retrodiction v2: Latin is the reward signal for "
    "structural and orthographic bridge search. Latin is used for scoring and "
    "proposal shaping but is never copied token-for-token into generated corpora."
)

CHAR_BIGRAM_TOP_N = 1500
CHAR_TRIGRAM_TOP_N = 2500
SUFFIX_TOP_N = 800
SUFFIX_LEN = 3
EPSILON = 1e-9

OPERATOR_NAMES = (
    "token_char_edit",
    "suffix_family_rewrite",
    "swap_bigram_order",
    "split_token",
    "merge_bigram",
    "sequence_span_rewrite",
)


@dataclass
class ReinforcedV2Config:
    """Tunable parameters for the relational v2 search."""

    num_sequences: int = 800
    max_proposals: int = 80
    max_accepted_stages: int = 18
    patience: int = 8
    seed: int = 42
    n_candidates: int = 6
    min_improvement: float = 0.001
    token_edit_attempts: int = 6
    suffix_candidate_samples: int = 8
    span_min_sequences: int = 2
    span_max_sequences: int = 5
    span_edit_min: int = 2
    span_edit_max: int = 4

    form_weight: float = 0.75
    coherence_weight: float = 0.05
    mutation_cost_weight: float = 0.005

    operator_weights: tuple[float, float, float, float, float, float] = (
        0.30,
        0.22,
        0.12,
        0.10,
        0.11,
        0.15,
    )

    def to_dict(self) -> dict:
        return {
            "num_sequences": self.num_sequences,
            "max_proposals": self.max_proposals,
            "max_accepted_stages": self.max_accepted_stages,
            "patience": self.patience,
            "seed": self.seed,
            "n_candidates": self.n_candidates,
            "min_improvement": self.min_improvement,
            "token_edit_attempts": self.token_edit_attempts,
            "suffix_candidate_samples": self.suffix_candidate_samples,
            "span_min_sequences": self.span_min_sequences,
            "span_max_sequences": self.span_max_sequences,
            "span_edit_min": self.span_edit_min,
            "span_edit_max": self.span_edit_max,
            "form_weight": self.form_weight,
            "coherence_weight": self.coherence_weight,
            "mutation_cost_weight": self.mutation_cost_weight,
            "operator_weights": list(self.operator_weights),
        }


def _build_sparse_profile(counter: Counter, top_n: int) -> dict[str, float]:
    top = counter.most_common(top_n)
    total = sum(count for _, count in top)
    if total == 0:
        return {}
    return {key: count / total for key, count in top}


def _extract_char_ngrams_from_sequences(sequences: list[list[str]], n: int) -> Counter:
    counter: Counter = Counter()
    for seq in sequences:
        for tok in seq:
            text = f"^{tok}$"
            if len(text) < n:
                continue
            for i in range(len(text) - n + 1):
                counter[text[i : i + n]] += 1
    return counter


def _extract_suffixes_from_sequences(sequences: list[list[str]], suffix_len: int = SUFFIX_LEN) -> Counter:
    counter: Counter = Counter()
    for seq in sequences:
        for tok in seq:
            if len(tok) >= suffix_len:
                counter[tok[-suffix_len:]] += 1
    return counter


def _sparse_profile_cosine(a: dict[str, float], b: dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    # Iterate over a only — cross-set terms (keys in b but not a) are zero,
    # so the union loop was equivalent but O(|a|+|b|) instead of O(|a|).
    # For score_token (|a|=~7, |b|=1018) this is a ~145x reduction.
    dot = sum(v * b.get(k, 0.0) for k, v in a.items())
    na = sum(v * v for v in a.values()) ** 0.5
    nb = sum(v * v for v in b.values()) ** 0.5
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(dot / (na * nb))


class LatinFormReference:
    """
    Orthographic/form reference derived from the Latin corpus.
    """

    def __init__(
        self,
        *,
        char_bigram_weight: float = 0.40,
        char_trigram_weight: float = 0.40,
        suffix_weight: float = 0.20,
    ) -> None:
        (
            self.char_bigram_weight,
            self.char_trigram_weight,
            self.suffix_weight,
        ) = self._normalize_form_weights(
            char_bigram_weight,
            char_trigram_weight,
            suffix_weight,
        )
        unlock_sequestration(V2_UNLOCK_REASON)
        try:
            self._load()
        finally:
            lock_sequestration()

    @staticmethod
    def _normalize_form_weights(
        char_bigram_weight: float,
        char_trigram_weight: float,
        suffix_weight: float,
    ) -> tuple[float, float, float]:
        weights = (
            float(char_bigram_weight),
            float(char_trigram_weight),
            float(suffix_weight),
        )
        if any(weight < 0.0 for weight in weights):
            raise ValueError("LatinFormReference weights must be non-negative.")
        total = sum(weights)
        if total <= 0.0:
            raise ValueError("LatinFormReference weights must sum to a positive value.")
        return tuple(weight / total for weight in weights)

    def _load(self) -> None:
        self._score_token_cache: dict[str, float] = {}
        latin_path = PROJECT_ROOT / "data" / "sequestered" / "latin" / "latin_tokens.json"
        with latin_path.open(encoding="utf-8") as fh:
            corpus = json.load(fh)

        sequences = corpus["sequences"][:50_000]
        self.char_bigram_profile = _build_sparse_profile(
            _extract_char_ngrams_from_sequences(sequences, 2),
            CHAR_BIGRAM_TOP_N,
        )
        self.char_trigram_profile = _build_sparse_profile(
            _extract_char_ngrams_from_sequences(sequences, 3),
            CHAR_TRIGRAM_TOP_N,
        )
        self.suffix_profile = _build_sparse_profile(
            _extract_suffixes_from_sequences(sequences, SUFFIX_LEN),
            SUFFIX_TOP_N,
        )

        char_counter = Counter()
        for seq in sequences:
            for tok in seq:
                char_counter.update(tok)

        self.mutation_chars = tuple(char_counter.keys())
        probs = np.array([char_counter[ch] for ch in self.mutation_chars], dtype=np.float64)
        self.mutation_char_probs = probs / probs.sum()

        suffix_keys = tuple(self.suffix_profile.keys()) or ("us", "um", "is")
        suffix_probs = np.array(
            [self.suffix_profile.get(sfx, 1.0 / len(suffix_keys)) for sfx in suffix_keys],
            dtype=np.float64,
        )
        self.sampleable_suffixes = suffix_keys
        self.sampleable_suffix_probs = suffix_probs / suffix_probs.sum()

        # Precompute reference norms once — they never change.
        # score_token uses these to avoid recomputing sum(v*v for v in ref.values())
        # on every call (that's O(|ref|) = O(1018-2500) per cosine call).
        self._bg_ref_norm  = sum(v * v for v in self.char_bigram_profile.values()) ** 0.5
        self._tg_ref_norm  = sum(v * v for v in self.char_trigram_profile.values()) ** 0.5
        self._sfx_ref_norm = sum(v * v for v in self.suffix_profile.values()) ** 0.5

        log.info(
            "Latin form reference loaded: %d char bigrams, %d char trigrams, %d suffixes",
            len(self.char_bigram_profile),
            len(self.char_trigram_profile),
            len(self.suffix_profile),
        )

    def sample_char(self, rng: np.random.Generator) -> str:
        idx = int(rng.choice(len(self.mutation_chars), p=self.mutation_char_probs))
        return self.mutation_chars[idx]

    def sample_suffix(self, rng: np.random.Generator) -> str:
        idx = int(rng.choice(len(self.sampleable_suffixes), p=self.sampleable_suffix_probs))
        return self.sampleable_suffixes[idx]

    def score_token(self, token: str) -> float:
        """Score a single token against the Latin reference profiles.

        Fast path: uses precomputed reference norms (_bg_ref_norm etc.) to
        avoid re-iterating the full Latin reference profile on every call.
        For a single token with ~5-20 char n-grams, this is O(~10) dict
        lookups instead of O(1018-2500) per cosine.
        """
        if not token:
            return 0.0
        cached = self._score_token_cache.get(token)
        if cached is not None:
            return cached

        # Extract char n-grams for this single token
        padded = f"^{token}$"
        n = len(padded)
        bg: dict[str, float] = {}
        tg: dict[str, float] = {}
        for i in range(n - 1):
            key = padded[i : i + 2]
            bg[key] = bg.get(key, 0.0) + 1.0
        for i in range(n - 2):
            key = padded[i : i + 3]
            tg[key] = tg.get(key, 0.0) + 1.0
        sfx: dict[str, float] = {}
        if len(token) >= SUFFIX_LEN:
            sfx[token[-SUFFIX_LEN:]] = 1.0

        # Normalize
        bg_total = sum(bg.values())
        if bg_total: bg = {k: v / bg_total for k, v in bg.items()}
        tg_total = sum(tg.values())
        if tg_total: tg = {k: v / tg_total for k, v in tg.items()}
        # sfx already has a single entry or is empty

        # Cosine against Latin reference — reference norms precomputed
        def _fast_cos(a: dict, ref: dict, ref_norm: float) -> float:
            if not a or ref_norm == 0.0:
                return 0.0
            dot = sum(v * ref.get(k, 0.0) for k, v in a.items())
            na = sum(v * v for v in a.values()) ** 0.5
            return float(dot / (na * ref_norm)) if na > 0.0 else 0.0

        bg_cos  = _fast_cos(bg,  self.char_bigram_profile,  self._bg_ref_norm)
        tg_cos  = _fast_cos(tg,  self.char_trigram_profile, self._tg_ref_norm)
        sfx_cos = _fast_cos(sfx, self.suffix_profile,       self._sfx_ref_norm)
        result = float(
            self.char_bigram_weight * bg_cos
            + self.char_trigram_weight * tg_cos
            + self.suffix_weight * sfx_cos
        )

        self._score_token_cache[token] = result
        return result

    def score(self, sequences: list[list[str]]) -> dict[str, float]:
        bg = _build_sparse_profile(_extract_char_ngrams_from_sequences(sequences, 2), CHAR_BIGRAM_TOP_N)
        tg = _build_sparse_profile(_extract_char_ngrams_from_sequences(sequences, 3), CHAR_TRIGRAM_TOP_N)
        sfx = _build_sparse_profile(_extract_suffixes_from_sequences(sequences, SUFFIX_LEN), SUFFIX_TOP_N)

        char_bigram_cos = _sparse_profile_cosine(bg, self.char_bigram_profile)
        char_trigram_cos = _sparse_profile_cosine(tg, self.char_trigram_profile)
        suffix_cos = _sparse_profile_cosine(sfx, self.suffix_profile)
        total = (
            self.char_bigram_weight * char_bigram_cos
            + self.char_trigram_weight * char_trigram_cos
            + self.suffix_weight * suffix_cos
        )
        return {
            "latin_form_score": float(total),
            "latin_char_bigram_cosine": float(char_bigram_cos),
            "latin_char_trigram_cosine": float(char_trigram_cos),
            "latin_suffix_cosine": float(suffix_cos),
        }


@dataclass
class ReinforcedV2StageRecord:
    stage_id: str
    source_language: str
    iteration: int
    proposal_index: int
    parent_stage_id: str | None
    mutation_operator: str
    mutation_details: str
    fingerprint_paths: dict
    artifact_paths: dict
    type_token_ratio: float
    bigram_coverage: float
    trigram_coverage: float
    bigram_entropy: float
    trigram_entropy: float
    structural_vector: list[float]
    latin_structural_score: float
    latin_form_score: float
    total_score: float
    scores: dict
    diagnostics: dict
    notes: str = ""
    flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "stage_id": self.stage_id,
            "source_language": self.source_language,
            "iteration": self.iteration,
            "proposal_index": self.proposal_index,
            "parent_stage_id": self.parent_stage_id,
            "mutation_operator": self.mutation_operator,
            "mutation_details": self.mutation_details,
            "fingerprint": {
                **self.fingerprint_paths,
                "type_token_ratio": self.type_token_ratio,
                "bigram_coverage": self.bigram_coverage,
                "trigram_coverage": self.trigram_coverage,
                "bigram_entropy": self.bigram_entropy,
                "trigram_entropy": self.trigram_entropy,
            },
            "artifacts": self.artifact_paths,
            "structural_vector": [round(v, 6) for v in self.structural_vector],
            "latin_structural_score": round(self.latin_structural_score, 6),
            "latin_form_score": round(self.latin_form_score, 6),
            "total_score": round(self.total_score, 6),
            "scores": self.scores,
            "diagnostics": self.diagnostics,
            "notes": self.notes,
            "flags": self.flags,
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, ensure_ascii=False, indent=2)


@dataclass
class CandidateState:
    sequences: list[list[str]]
    operator: str
    details: str
    mutation_cost: float
    structural_vector: np.ndarray
    latin_structural_score: float
    latin_form_score: float
    form_details: dict
    total_score: float
    scores: dict
    diagnostics: dict
    type_token_ratio: float
    bigram_coverage: float
    trigram_coverage: float
    bigram_profile: dict[str, float]
    trigram_profile: dict[str, float]


class RelationalReinforcedRetrodictionEngine:
    """
    A multi-scale reinforced search over actual corpora.
    """

    def __init__(
        self,
        language: str,
        source_sequences: list[list[str]],
        latin_structural_ref: LatinReference,
        latin_form_ref: LatinFormReference,
        config: ReinforcedV2Config | None = None,
        output_dir: Path | None = None,
        references: ReferenceSet | None = None,
    ) -> None:
        self.language = language
        self.source_sequences = source_sequences
        self.latin_structural_ref = latin_structural_ref
        self.latin_form_ref = latin_form_ref
        self.config = config or ReinforcedV2Config()
        self.lang_code = LANG_CODES.get(language, language[:3].upper())
        self._references = references or ReferenceSet()

        if output_dir is None:
            output_dir = RETRODICTION_DIR / language / "v2"
        self.output_dir = output_dir
        self.records_dir = output_dir / "records"
        self.matrices_dir = output_dir / "matrices"
        self.corpora_dir = output_dir / "corpora"
        self.preview_dir = output_dir / "previews"
        self.records_dir.mkdir(parents=True, exist_ok=True)
        self.matrices_dir.mkdir(parents=True, exist_ok=True)
        self.corpora_dir.mkdir(parents=True, exist_ok=True)
        self.preview_dir.mkdir(parents=True, exist_ok=True)

    def _stage_id(self, iteration: int) -> str:
        return f"{self.lang_code}_v2_{iteration:03d}"

    def _sample_initial_corpus(self, rng: np.random.Generator) -> list[list[str]]:
        n = min(self.config.num_sequences, len(self.source_sequences))
        idx = rng.choice(len(self.source_sequences), size=n, replace=False)
        return [list(self.source_sequences[int(i)]) for i in idx]

    def _token_counts(self, sequences: list[list[str]]) -> Counter:
        return Counter(tok for seq in sequences for tok in seq)

    def _bigram_counts(self, sequences: list[list[str]]) -> Counter:
        return extract_ngrams(sequences, 2)

    def _clone_sequences(self, sequences: list[list[str]]) -> list[list[str]]:
        return [list(seq) for seq in sequences]

    def _splice_sequence_span(
        self,
        sequences: list[list[str]],
        start: int,
        replacement_span: list[list[str]],
    ) -> list[list[str]]:
        new_sequences = self._clone_sequences(sequences)
        new_sequences[start : start + len(replacement_span)] = self._clone_sequences(replacement_span)
        return new_sequences

    def _splice_sequence_span_sparse(
        self,
        sequences: list[list[str]],
        start: int,
        replacement_span: list[list[str]],
    ) -> dict[int, list[str]]:
        changed_sequences: dict[int, list[str]] = {}
        for offset, seq in enumerate(replacement_span):
            seq_idx = start + offset
            if sequences[seq_idx] != seq:
                changed_sequences[seq_idx] = list(seq)
        return changed_sequences

    def _apply_token_rewrite(
        self,
        sequences: list[list[str]],
        rewrite_map: dict[str, str],
    ) -> tuple[list[list[str]], int]:
        new_sequences = self._clone_sequences(sequences)
        replacements = 0
        for seq in new_sequences:
            for j, tok in enumerate(seq):
                if tok in rewrite_map:
                    seq[j] = rewrite_map[tok]
                    replacements += 1
        return new_sequences, replacements

    def _apply_token_rewrite_sparse(
        self,
        sequences: list[list[str]],
        rewrite_map: dict[str, str],
    ) -> tuple[dict[int, list[str]], int]:
        changed_sequences: dict[int, list[str]] = {}
        replacements = 0
        if not rewrite_map:
            return changed_sequences, replacements

        for seq_idx, seq in enumerate(sequences):
            new_seq: list[str] | None = None
            local_changes = 0
            for tok_idx, tok in enumerate(seq):
                new_tok = rewrite_map.get(tok)
                if new_tok is None:
                    if new_seq is not None:
                        new_seq.append(tok)
                    continue
                if new_seq is None:
                    new_seq = list(seq[:tok_idx])
                new_seq.append(new_tok)
                local_changes += 1
            if new_seq is not None:
                changed_sequences[seq_idx] = new_seq
                replacements += local_changes

        return changed_sequences, replacements

    def _split_token_sparse(
        self,
        sequences: list[list[str]],
        tok: str,
        left: str,
        right: str,
    ) -> tuple[dict[int, list[str]], int]:
        changed_sequences: dict[int, list[str]] = {}
        replacements = 0
        for seq_idx, seq in enumerate(sequences):
            new_seq: list[str] | None = None
            for tok_idx, item in enumerate(seq):
                if item == tok:
                    if new_seq is None:
                        new_seq = list(seq[:tok_idx])
                    new_seq.extend([left, right])
                    replacements += 1
                else:
                    if new_seq is not None:
                        new_seq.append(item)
            if new_seq is not None:
                changed_sequences[seq_idx] = new_seq
        return changed_sequences, replacements

    def _random_edit_token_form(self, tok: str, rng: np.random.Generator) -> str:
        if not tok:
            return tok

        mode = str(rng.choice(["substitute", "delete", "insert"], p=[0.5, 0.2, 0.3]))
        chars = list(tok)
        if mode == "substitute" and chars:
            pos = int(rng.integers(0, len(chars)))
            chars[pos] = self.latin_form_ref.sample_char(rng)
        elif mode == "delete" and len(chars) > 2:
            pos = int(rng.integers(0, len(chars)))
            del chars[pos]
        else:
            pos = int(rng.integers(0, len(chars) + 1))
            chars.insert(pos, self.latin_form_ref.sample_char(rng))

        new_tok = "".join(chars).strip()
        return tok if len(new_tok) < 2 else new_tok

    def _edit_token_form(self, tok: str, rng: np.random.Generator) -> str:
        base_score = self.latin_form_ref.score_token(tok)
        best_tok = tok
        best_score = base_score

        for _ in range(self.config.token_edit_attempts):
            candidate = self._random_edit_token_form(tok, rng)
            if candidate == tok:
                continue
            candidate_score = self.latin_form_ref.score_token(candidate)
            if candidate_score > best_score:
                best_tok = candidate
                best_score = candidate_score

        return best_tok

    def _mutate_token_char_edit(
        self,
        sequences: list[list[str]],
        token_counts: Counter,
        rng: np.random.Generator,
    ) -> tuple[list[list[str]] | None, str, float]:
        candidates = [(tok, count) for tok, count in token_counts.items() if len(tok) >= 3]
        if not candidates:
            return None, "no eligible token", 0.0

        weights = np.array([count * max(count, 1) * max(len(tok), 1) for tok, count in candidates], dtype=np.float64)
        weights /= weights.sum()
        idx = int(rng.choice(len(candidates), p=weights))
        tok = candidates[idx][0]
        new_tok = self._edit_token_form(tok, rng)
        if new_tok == tok:
            return None, f"token {tok} unchanged", 0.0

        new_sequences, affected = self._apply_token_rewrite(sequences, {tok: new_tok})
        if affected == 0:
            return None, f"token {tok} had no occurrences", 0.0
        return new_sequences, f"{tok} -> {new_tok} ({affected} occurrences)", 0.25

    def _mutate_suffix_family(
        self,
        sequences: list[list[str]],
        token_counts: Counter,
        rng: np.random.Generator,
    ) -> tuple[list[list[str]] | None, str, float]:
        tokens = [tok for tok in token_counts if len(tok) >= 4]
        if not tokens:
            return None, "no suffix-family candidates", 0.0

        target = str(rng.choice(tokens))
        suffix_len = int(rng.choice([2, 3, 4], p=[0.35, 0.45, 0.20]))
        suffix_len = min(suffix_len, len(target) - 1)
        suffix = target[-suffix_len:]
        family = [tok for tok in token_counts if tok.endswith(suffix) and len(tok) > suffix_len]
        if len(family) < 2:
            return None, f"suffix family too small for {suffix}", 0.0

        base_score = sum(
            token_counts[tok] * self.latin_form_ref.score_token(tok)
            for tok in family
        )
        best_suffix = suffix
        best_score = base_score

        for _ in range(self.config.suffix_candidate_samples):
            candidate_suffix = self.latin_form_ref.sample_suffix(rng)
            if candidate_suffix == suffix:
                candidate_suffix = self._random_edit_token_form(suffix, rng)
            if candidate_suffix == suffix or len(candidate_suffix) < 1:
                continue
            candidate_score = sum(
                token_counts[tok] * self.latin_form_ref.score_token(tok[:-suffix_len] + candidate_suffix)
                for tok in family
            )
            if candidate_score > best_score:
                best_suffix = candidate_suffix
                best_score = candidate_score

        if best_suffix == suffix:
            return None, f"suffix {suffix} unchanged", 0.0

        rewrite_map = {tok: tok[:-suffix_len] + best_suffix for tok in family}
        new_sequences, affected = self._apply_token_rewrite(sequences, rewrite_map)
        if affected == 0:
            return None, f"suffix family {suffix} had no affected tokens", 0.0
        cost = 0.5 + 0.005 * len(rewrite_map)
        details = f"{suffix} -> {best_suffix} across {len(rewrite_map)} token types, {affected} occurrences"
        return new_sequences, details, cost

    def _mutate_swap_bigram_order(
        self,
        sequences: list[list[str]],
        bigram_counts: Counter,
        rng: np.random.Generator,
    ) -> tuple[list[list[str]] | None, str, float]:
        candidates = [(gram, count) for gram, count in bigram_counts.items() if count >= 2 and gram[0] != gram[1]]
        if not candidates:
            return None, "no swappable bigrams", 0.0

        weights = np.array([count for _, count in candidates], dtype=np.float64)
        weights /= weights.sum()
        idx = int(rng.choice(len(candidates), p=weights))
        (a, b), _ = candidates[idx]

        new_sequences = self._clone_sequences(sequences)
        swaps = 0
        for seq in new_sequences:
            i = 0
            while i < len(seq) - 1:
                if seq[i] == a and seq[i + 1] == b and rng.random() < 0.5:
                    seq[i], seq[i + 1] = seq[i + 1], seq[i]
                    swaps += 1
                    i += 2
                else:
                    i += 1
        if swaps == 0:
            return None, f"no swaps applied for {a} {b}", 0.0
        return new_sequences, f"swapped {a} {b} in {swaps} positions", 0.2

    def _mutate_split_token(
        self,
        sequences: list[list[str]],
        token_counts: Counter,
        rng: np.random.Generator,
    ) -> tuple[list[list[str]] | None, str, float]:
        tokens = [(tok, count) for tok, count in token_counts.items() if len(tok) >= 6]
        if not tokens:
            return None, "no splittable tokens", 0.0

        weights = np.array([count * len(tok) for tok, count in tokens], dtype=np.float64)
        weights /= weights.sum()
        idx = int(rng.choice(len(tokens), p=weights))
        tok = tokens[idx][0]
        base_score = self.latin_form_ref.score_token(tok)
        best_parts: tuple[str, str] | None = None
        best_score = base_score

        for split_at in range(2, len(tok) - 1):
            left, right = tok[:split_at], tok[split_at:]
            if len(left) < 2 or len(right) < 2:
                continue
            candidate_score = (
                self.latin_form_ref.score_token(left)
                + self.latin_form_ref.score_token(right)
            )
            if candidate_score > best_score:
                best_parts = (left, right)
                best_score = candidate_score

        if best_parts is None:
            return None, f"no beneficial split for {tok}", 0.0

        left, right = best_parts

        new_sequences = self._clone_sequences(sequences)
        replacements = 0
        for i, seq in enumerate(new_sequences):
            new_seq = []
            for item in seq:
                if item == tok:
                    new_seq.extend([left, right])
                    replacements += 1
                else:
                    new_seq.append(item)
            new_sequences[i] = new_seq

        if replacements == 0:
            return None, f"token {tok} not split", 0.0
        return new_sequences, f"{tok} -> {left} + {right} ({replacements} occurrences)", 0.4

    def _mutate_merge_bigram(
        self,
        sequences: list[list[str]],
        bigram_counts: Counter,
        rng: np.random.Generator,
    ) -> tuple[list[list[str]] | None, str, float]:
        candidates = [
            (gram, count)
            for gram, count in bigram_counts.items()
            if count >= 2 and len(gram[0]) >= 2 and len(gram[1]) >= 2
        ]
        if not candidates:
            return None, "no mergeable bigrams", 0.0

        weights = np.array([count for _, count in candidates], dtype=np.float64)
        weights /= weights.sum()
        sampled = rng.choice(len(candidates), size=min(12, len(candidates)), replace=False, p=weights)

        best_pair: tuple[str, str] | None = None
        best_merged = ""
        best_delta = 0.0
        for raw_idx in np.atleast_1d(sampled):
            idx = int(raw_idx)
            (a, b), _ = candidates[idx]
            merged = a + b
            delta = self.latin_form_ref.score_token(merged) - (
                0.5 * self.latin_form_ref.score_token(a)
                + 0.5 * self.latin_form_ref.score_token(b)
            )
            if delta > best_delta:
                best_pair = (a, b)
                best_merged = merged
                best_delta = delta

        if best_pair is None:
            return None, "no beneficial bigram merge", 0.0

        a, b = best_pair
        merged = best_merged

        new_sequences = self._clone_sequences(sequences)
        merges = 0
        for s_idx, seq in enumerate(new_sequences):
            out = []
            i = 0
            while i < len(seq):
                if i < len(seq) - 1 and seq[i] == a and seq[i + 1] == b and rng.random() < 0.5:
                    out.append(merged)
                    merges += 1
                    i += 2
                else:
                    out.append(seq[i])
                    i += 1
            new_sequences[s_idx] = out

        if merges == 0:
            return None, f"no merges applied for {a} {b}", 0.0
        return new_sequences, f"{a} + {b} -> {merged} ({merges} merges)", 0.45

    def _mutate_sequence_span_rewrite(
        self,
        sequences: list[list[str]],
        rng: np.random.Generator,
    ) -> tuple[list[list[str]] | None, str, float]:
        min_span = max(1, self.config.span_min_sequences)
        max_span = min(self.config.span_max_sequences, len(sequences))
        if len(sequences) < min_span or max_span < min_span:
            return None, "not enough sequences for span rewrite", 0.0

        span_len = int(rng.integers(min_span, max_span + 1))
        start = int(rng.integers(0, len(sequences) - span_len + 1))
        span = self._clone_sequences(sequences[start : start + span_len])
        mutated_span, detail_text, accumulated_cost = self._rewrite_sequence_span_local(span, rng)
        if mutated_span is None:
            return None, f"span[{start}:{start + span_len}] unchanged", 0.0

        new_sequences = self._splice_sequence_span(sequences, start, mutated_span)
        total_cost = 0.75 + accumulated_cost + 0.08 * max(span_len - 1, 0)
        return new_sequences, f"span[{start}:{start + span_len}] {detail_text}", total_cost

    def _rewrite_sequence_span_local(
        self,
        span: list[list[str]],
        rng: np.random.Generator,
    ) -> tuple[list[list[str]] | None, str, float]:
        original_span = self._clone_sequences(span)
        working_span = self._clone_sequences(span)

        sub_ops = (
            "token_char_edit",
            "suffix_family_rewrite",
            "swap_bigram_order",
            "split_token",
            "merge_bigram",
            "sequence_order",
        )
        sub_weights = np.array([0.28, 0.24, 0.16, 0.12, 0.08, 0.12], dtype=np.float64)
        sub_weights /= sub_weights.sum()

        desired_edits = int(rng.integers(self.config.span_edit_min, self.config.span_edit_max + 1))
        details_parts: list[str] = []
        accumulated_cost = 0.0

        for _ in range(desired_edits):
            token_counts = self._token_counts(working_span)
            bigram_counts = self._bigram_counts(working_span)
            sub_op = str(rng.choice(sub_ops, p=sub_weights))

            if sub_op == "token_char_edit":
                mutated, details, cost = self._mutate_token_char_edit(working_span, token_counts, rng)
            elif sub_op == "suffix_family_rewrite":
                mutated, details, cost = self._mutate_suffix_family(working_span, token_counts, rng)
            elif sub_op == "swap_bigram_order":
                mutated, details, cost = self._mutate_swap_bigram_order(working_span, bigram_counts, rng)
            elif sub_op == "split_token":
                mutated, details, cost = self._mutate_split_token(working_span, token_counts, rng)
            elif sub_op == "merge_bigram":
                mutated, details, cost = self._mutate_merge_bigram(working_span, bigram_counts, rng)
            else:
                if len(working_span) < 2:
                    mutated, details, cost = None, "span too short for sequence reorder", 0.0
                else:
                    mutated = self._clone_sequences(working_span)
                    if rng.random() < 0.5:
                        idx = int(rng.integers(0, len(mutated) - 1))
                        mutated[idx], mutated[idx + 1] = mutated[idx + 1], mutated[idx]
                        details = f"swapped sequence {idx} with {idx + 1}"
                    else:
                        shift = int(rng.integers(1, len(mutated)))
                        mutated = mutated[shift:] + mutated[:shift]
                        details = f"rotated span by {shift}"
                    cost = 0.30

            if mutated is None or mutated == working_span:
                continue

            working_span = mutated
            accumulated_cost += cost
            details_parts.append(f"{sub_op}:{details}")

        if working_span == original_span:
            return None, "multi-sequence rewrite", 0.0

        detail_text = "; ".join(details_parts[:4]) if details_parts else "multi-sequence rewrite"
        return working_span, detail_text, accumulated_cost

    def _choose_operator(self, rng: np.random.Generator) -> str:
        weights = np.array(self.config.operator_weights, dtype=np.float64)
        weights /= weights.sum()
        idx = int(rng.choice(len(OPERATOR_NAMES), p=weights))
        return OPERATOR_NAMES[idx]

    def _mutate_candidate(
        self,
        sequences: list[list[str]],
        rng: np.random.Generator,
    ) -> tuple[list[list[str]] | None, str, str, float]:
        token_counts = self._token_counts(sequences)
        bigram_counts = self._bigram_counts(sequences)

        for _ in range(8):
            operator = self._choose_operator(rng)
            if operator == "token_char_edit":
                mutated, details, cost = self._mutate_token_char_edit(sequences, token_counts, rng)
            elif operator == "suffix_family_rewrite":
                mutated, details, cost = self._mutate_suffix_family(sequences, token_counts, rng)
            elif operator == "swap_bigram_order":
                mutated, details, cost = self._mutate_swap_bigram_order(sequences, bigram_counts, rng)
            elif operator == "split_token":
                mutated, details, cost = self._mutate_split_token(sequences, token_counts, rng)
            elif operator == "sequence_span_rewrite":
                mutated, details, cost = self._mutate_sequence_span_rewrite(sequences, rng)
            else:
                mutated, details, cost = self._mutate_merge_bigram(sequences, bigram_counts, rng)

            if mutated is not None:
                return mutated, operator, details, cost

        return None, "none", "no valid mutation generated", 0.0

    def _evaluate_sequences(
        self,
        sequences: list[list[str]],
        mutation_cost: float,
    ) -> CandidateState:
        from src.ingest.tokenize import corpus_stats

        bg_counts = extract_ngrams(sequences, 2)
        tg_counts = extract_ngrams(sequences, 3)
        bg_profile = build_profile(bg_counts, 5000)
        tg_profile = build_profile(tg_counts, 5000)
        vec = structural_vector(sequences, bg_profile, tg_profile)
        structural_score = self.latin_structural_ref.score(vec)
        form_details = self.latin_form_ref.score(sequences)
        form_score = form_details["latin_form_score"]
        scores = self._references.score(sequences, bg_profile, tg_profile)
        diagnostics = self._references.coherence_from_vector(vec)
        stats = corpus_stats(sequences)

        total_score = (
            structural_score
            + self.config.form_weight * form_score
            + self.config.coherence_weight * diagnostics["language_likeness_margin"]
            - self.config.mutation_cost_weight * mutation_cost
        )

        return CandidateState(
            sequences=sequences,
            operator="",
            details="",
            mutation_cost=mutation_cost,
            structural_vector=vec,
            latin_structural_score=structural_score,
            latin_form_score=form_score,
            form_details=form_details,
            total_score=float(total_score),
            scores=scores,
            diagnostics=diagnostics,
            type_token_ratio=stats["type_token_ratio"],
            bigram_coverage=float(vec[1]),
            trigram_coverage=float(vec[2]),
            bigram_profile=bg_profile,
            trigram_profile=tg_profile,
        )

    def _save_stage(
        self,
        candidate: CandidateState,
        stage_id: str,
        iteration: int,
        proposal_index: int,
        parent_stage_id: str | None,
        mutation_operator: str,
        mutation_details: str,
        save_dense_matrices: bool = True,
    ) -> ReinforcedV2StageRecord:
        fp_paths, _, _, ttr, bg_cov, tg_cov, bg_ent, tg_ent = _fingerprint_sequences(
            stage_id,
            candidate.sequences,
            self.matrices_dir,
            save_dense_matrices=save_dense_matrices,
        )
        artifact_paths = _save_stage_corpus(stage_id, candidate.sequences, self.corpora_dir, self.preview_dir)

        record = ReinforcedV2StageRecord(
            stage_id=stage_id,
            source_language=self.language,
            iteration=iteration,
            proposal_index=proposal_index,
            parent_stage_id=parent_stage_id,
            mutation_operator=mutation_operator,
            mutation_details=mutation_details,
            fingerprint_paths=fp_paths,
            artifact_paths=artifact_paths,
            type_token_ratio=ttr,
            bigram_coverage=bg_cov,
            trigram_coverage=tg_cov,
            bigram_entropy=bg_ent,
            trigram_entropy=tg_ent,
            structural_vector=candidate.structural_vector.tolist(),
            latin_structural_score=candidate.latin_structural_score,
            latin_form_score=candidate.latin_form_score,
            total_score=candidate.total_score,
            scores=candidate.scores,
            diagnostics={
                **candidate.diagnostics,
                **candidate.form_details,
                "mutation_cost": candidate.mutation_cost,
            },
            notes="",
        )
        record.save(self.records_dir / f"{stage_id}.json")
        return record

    def run(self) -> list[ReinforcedV2StageRecord]:
        cfg = self.config
        rng = np.random.default_rng(cfg.seed)

        log.info(
            "Relational reinforced v2: language=%s, num_sequences=%d, proposals=%d, candidates=%d",
            self.language, cfg.num_sequences, cfg.max_proposals, cfg.n_candidates,
        )

        current_sequences = self._sample_initial_corpus(rng)
        current = self._evaluate_sequences(current_sequences, mutation_cost=0.0)
        records: list[ReinforcedV2StageRecord] = []

        stage_index = 0
        current_stage_id = self._stage_id(stage_index)
        seed_record = self._save_stage(
            current,
            current_stage_id,
            iteration=stage_index,
            proposal_index=0,
            parent_stage_id=None,
            mutation_operator="seed",
            mutation_details="initial sampled source baseline",
        )
        records.append(seed_record)

        stagnation = 0
        accepted_ops: Counter = Counter()
        halt_reason = "max_proposals"
        proposals_attempted = 0

        for proposal_index in range(1, cfg.max_proposals + 1):
            proposals_attempted = proposal_index
            best_candidate: CandidateState | None = None
            best_operator = "none"
            best_details = ""

            for _ in range(cfg.n_candidates):
                mutated, operator, details, mutation_cost = self._mutate_candidate(current.sequences, rng)
                if mutated is None:
                    continue
                candidate = self._evaluate_sequences(mutated, mutation_cost=mutation_cost)
                candidate.operator = operator
                candidate.details = details
                if best_candidate is None or candidate.total_score > best_candidate.total_score:
                    best_candidate = candidate
                    best_operator = operator
                    best_details = details

            if best_candidate is None:
                stagnation += 1
                if stagnation >= cfg.patience:
                    halt_reason = "no_valid_mutations"
                    break
                continue

            improvement = best_candidate.total_score - current.total_score
            if improvement > cfg.min_improvement:
                stage_index += 1
                parent_stage_id = current_stage_id
                current = best_candidate
                current_stage_id = self._stage_id(stage_index)
                record = self._save_stage(
                    current,
                    current_stage_id,
                    iteration=stage_index,
                    proposal_index=proposal_index,
                    parent_stage_id=parent_stage_id,
                    mutation_operator=best_operator,
                    mutation_details=best_details,
                )
                records.append(record)
                accepted_ops[best_operator] += 1
                stagnation = 0

                log.info(
                    "V2 %s: total=%.4f struct=%.4f form=%.4f coherence=%s op=%s",
                    current_stage_id,
                    current.total_score,
                    current.latin_structural_score,
                    current.latin_form_score,
                    current.diagnostics["coherence_label"],
                    best_operator,
                )

                if stage_index + 1 >= cfg.max_accepted_stages:
                    halt_reason = "max_accepted_stages"
                    break
            else:
                stagnation += 1
                if stagnation >= cfg.patience:
                    halt_reason = "stable"
                    if records:
                        records[-1].flags.append("stable")
                        records[-1].save(self.records_dir / f"{records[-1].stage_id}.json")
                    break

        self._save_summary(records, cfg, proposals_attempted, halt_reason, accepted_ops)
        return records

    def _save_summary(
        self,
        records: list[ReinforcedV2StageRecord],
        cfg: ReinforcedV2Config,
        proposals_attempted: int,
        halt_reason: str,
        accepted_ops: Counter,
    ) -> None:
        best_record = max(records, key=lambda r: r.total_score) if records else None
        summary = {
            "language": self.language,
            "algorithm": "relational_v2",
            "config": cfg.to_dict(),
            "total_stages": len(records),
            "accepted_mutation_stages": max(len(records) - 1, 0),
            "proposals_attempted": proposals_attempted,
            "halt_reason": halt_reason,
            "accepted_operator_counts": dict(accepted_ops),
            "final_stage_id": records[-1].stage_id if records else None,
            "final_total_score": records[-1].total_score if records else None,
            "final_latin_structural_score": records[-1].latin_structural_score if records else None,
            "final_latin_form_score": records[-1].latin_form_score if records else None,
            "final_coherence_label": records[-1].diagnostics.get("coherence_label") if records else None,
            "best_stage_id": best_record.stage_id if best_record else None,
            "best_total_score": best_record.total_score if best_record else None,
            "best_latin_structural_score": best_record.latin_structural_score if best_record else None,
            "best_latin_form_score": best_record.latin_form_score if best_record else None,
            "best_corpus_json": best_record.artifact_paths.get("corpus_json") if best_record else None,
            "best_preview_txt": best_record.artifact_paths.get("preview_txt") if best_record else None,
            "stages": [r.to_dict() for r in records],
        }
        path = self.output_dir / "run_summary.json"
        with path.open("w", encoding="utf-8") as fh:
            json.dump(summary, fh, ensure_ascii=False, indent=2)
        log.info("Saved reinforced v2 summary to %s", path)


def run(
    language: str,
    config: ReinforcedV2Config | None = None,
    input_path: Path | None = None,
) -> list[ReinforcedV2StageRecord]:
    if input_path is None:
        input_path = PROCESSED_DIR / "romance" / f"{language}_tokens.json"

    log.info("Loading source corpus from %s", input_path)
    with input_path.open(encoding="utf-8") as fh:
        corpus = json.load(fh)
    sequences = corpus["sequences"]
    log.info("Loaded %d sequences", len(sequences))

    cfg = config or ReinforcedV2Config()
    latin_structural_ref = LatinReference()
    latin_form_ref = LatinFormReference()
    references = ReferenceSet()

    engine = RelationalReinforcedRetrodictionEngine(
        language=language,
        source_sequences=sequences,
        latin_structural_ref=latin_structural_ref,
        latin_form_ref=latin_form_ref,
        config=cfg,
        references=references,
    )
    return engine.run()
