"""
Reinforcement-Guided Retrodiction Engines
==========================================
Purpose:
    Two retrodiction engines that use Latin as an explicit reinforcement signal.
    Both move FROM a modern Romance language TOWARD Latin, recording every
    bridge stage. The path is the finding — not the destination.

    The scientific question is not "can we get to Latin?" (we can, by construction)
    but "what path does each algorithm take, and how does that path compare to
    attested intermediate languages (Old French, Carolingian Latin, Vulgar Latin)?"

Single-blind design:
    Latin is the reward signal — the "parent in the room." This is intentional
    and methodologically sound. A baby is not double-blind with respect to its
    parents' language. The interesting result is the trajectory, not the arrival.

Two algorithms for comparison:
    Option A — Stochastic (baby babble):
        At each iteration, generate N random perturbations of the current bigram
        model. Score each against the Latin reference vector. Keep the best.
        True reinforcement learning: random exploration, selection by reward.
        The model finds its own path through random variation + selection.

    Option B — Directed gradient:
        At each iteration, mix the current bigram transition matrix one step
        toward the Latin bigram transition matrix. Deterministic. Greedy.
        Finds the shortest path in transition-matrix space.

Comparison:
    Both produce BridgeStageRecord sequences in the same format. Comparing
    the two paths identifies where the space is wide (A and B diverge) vs
    narrow (A and B agree — the gradient was the only viable direction).

Scoring:
    Euclidean distance in structural feature space:
        score = -||vec_current - vec_latin||
    Higher score = closer to Latin. Used for candidate selection in Option A
    and as a convergence monitor in Option B.

Output:
    data/retrodiction/{language}/stochastic/   — Option A
    data/retrodiction/{language}/gradient/     — Option B
"""

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.retrodiction.generate import BigramModel
from src.retrodiction.similarity import (
    ReferenceSet,
    profile_entropy,
    structural_vector,
    top_k_coverage,
)
from src.fingerprint import cooccurrence, positional, ngram
from src.sequester.guard import unlock_sequestration, lock_sequestration

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MATRICES_DIR = PROJECT_ROOT / "data" / "matrices"
RETRODICTION_DIR = PROJECT_ROOT / "data" / "retrodiction"

LANG_CODES = {
    "french": "FR",
    "italian": "IT",
    "spanish": "ES",
    "romanian": "RO",
    "occitan": "OC",
    "genoese": "LIJ",
}

REWARD_FEATURE_NAMES = (
    "type_token_ratio",
    "bigram_coverage",
    "trigram_coverage",
)
REWARD_SCORE_SCALE = 5.0
PREVIEW_SEQUENCE_COUNT = 40

UNLOCK_REASON = (
    "Phase 3 reinforcement retrodiction: Latin is the reward signal and "
    "gradient target. Loading Latin structural reference data for scoring and "
    "transition guidance. Latin sequences are never copied token-for-token "
    "into the generated intermediate corpora."
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ReinforcedConfig:
    """Parameters shared by both engine variants."""
    num_sequences: int = 2000       # synthetic sequences per stage
    max_iterations: int = 200       # hard upper limit
    stability_threshold: float = 0.002   # L2 delta in score to declare stability
    seed: int = 42

    # Stochastic-only
    n_candidates: int = 20          # candidates per generation
    noise_scale: float = 0.3        # Gaussian noise scale in log-prob space

    # Gradient-only
    alpha: float = 0.05             # mixing rate toward Latin per step

    def to_dict(self) -> dict:
        return {
            "num_sequences": self.num_sequences,
            "max_iterations": self.max_iterations,
            "stability_threshold": self.stability_threshold,
            "seed": self.seed,
            "n_candidates": self.n_candidates,
            "noise_scale": self.noise_scale,
            "alpha": self.alpha,
        }


# ---------------------------------------------------------------------------
# Latin reference
# ---------------------------------------------------------------------------

class LatinReference:
    """
    Loads the Latin bigram model and structural vector once.
    Requires sequestration unlock — documented at class level.
    """

    def __init__(self) -> None:
        unlock_sequestration(UNLOCK_REASON)
        try:
            self._load()
        finally:
            lock_sequestration()

    def _load(self) -> None:
        latin_path = PROJECT_ROOT / "data" / "sequestered" / "latin" / "latin_tokens.json"
        ngram_meta_path = MATRICES_DIR / "latin_ngram_meta.json"

        log.info("Loading Latin reference corpus...")
        with latin_path.open(encoding="utf-8") as fh:
            corpus = json.load(fh)

        # Sample for structural vector — full corpus is 897K sequences,
        # 50K is sufficient for stable feature estimates
        sequences = corpus["sequences"][:50_000]

        with ngram_meta_path.open(encoding="utf-8") as fh:
            ngram_meta = json.load(fh)

        self.vec = structural_vector(sequences, ngram_meta["bigrams"], ngram_meta["trigrams"])
        self.reward_vec = self.vec[:3].copy()
        self.score_scale = REWARD_SCORE_SCALE
        log.info(
            "Latin reference vector: TTR=%.4f bg_cov=%.4f tg_cov=%.4f log_seq=%.4f",
            self.vec[0], self.vec[1], self.vec[2], self.vec[3],
        )
        log.info(
            "Latin reward subspace: TTR=%.4f bg_cov=%.4f tg_cov=%.4f (log_seq excluded; source lengths are fixed during generation)",
            self.reward_vec[0], self.reward_vec[1], self.reward_vec[2],
        )

        # Build Latin bigram model for directed gradient
        log.info("Building Latin bigram model (for gradient engine)...")
        self.model = BigramModel.from_sequences(sequences)
        log.info("Latin bigram model: V=%d", self.model.V)

    def score(self, vec: np.ndarray) -> float:
        """
        Score a structural vector against Latin.
        Returns negative Euclidean distance in the trainable reward subspace:
            [type_token_ratio, bigram_coverage, trigram_coverage]

        The fourth structural feature, log mean sequence length, is excluded from
        the reward because the generator preserves the source sentence-length
        distribution and cannot optimize that dimension directly.
        """
        reward_vec = getattr(self, "reward_vec", np.asarray(self.vec)[:3])
        score_scale = getattr(self, "score_scale", REWARD_SCORE_SCALE)
        candidate = np.asarray(vec, dtype=np.float64)[:reward_vec.shape[0]]
        return -float(score_scale * np.linalg.norm(candidate - reward_vec))


# ---------------------------------------------------------------------------
# Bridge stage record
# ---------------------------------------------------------------------------

@dataclass
class ReinforcedStageRecord:
    stage_id: str
    source_language: str
    algorithm: str          # "stochastic" or "gradient"
    iteration: int
    fingerprint_paths: dict
    type_token_ratio: float
    bigram_coverage: float
    trigram_coverage: float
    structural_vector: list[float]
    latin_score: float      # -||vec - vec_latin|| (higher = closer)
    scores: dict
    artifact_paths: dict = field(default_factory=dict)
    bigram_entropy: float = 0.0
    trigram_entropy: float = 0.0
    diagnostics: dict = field(default_factory=dict)
    notes: str = ""
    flags: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "stage_id": self.stage_id,
            "source_language": self.source_language,
            "algorithm": self.algorithm,
            "iteration": self.iteration,
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
            "latin_score": round(self.latin_score, 6),
            "scores": self.scores,
            "diagnostics": self.diagnostics,
            "notes": self.notes,
            "flags": self.flags,
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Shared fingerprinting helper
# ---------------------------------------------------------------------------

def _fingerprint_sequences(
    stage_id: str,
    sequences: list[list[str]],
    matrices_dir: Path,
    save_dense_matrices: bool = True,
) -> tuple[dict, dict, dict, float, float, float, float, float]:
    """
    Fingerprint a generated corpus. Returns:
        (paths, bigram_profile, trigram_profile, ttr, bg_cov, tg_cov, bg_ent, tg_ent)

    When save_dense_matrices=False, skips the cooccurrence and positional .npy
    files (~48MB each). Only the ngram_meta.json is written. The corpus JSON
    is always saved separately via _save_stage_corpus. Everything needed for
    checkpoint_compare.py is preserved; the dense matrices are recomputable
    from the corpus JSON if ever needed.
    """
    if save_dense_matrices:
        cooccurrence.run_from_sequences(stage_id, sequences, output_dir=matrices_dir)
        positional.run_from_sequences(stage_id, sequences, output_dir=matrices_dir)
    bigram_profile, trigram_profile, ttr = ngram.run_from_sequences(
        stage_id, sequences, output_dir=matrices_dir
    )
    bg_cov = top_k_coverage(bigram_profile)
    tg_cov = top_k_coverage(trigram_profile)
    bg_ent = profile_entropy(bigram_profile)
    tg_ent = profile_entropy(trigram_profile)
    paths = {
        "ngram_meta": str(matrices_dir / f"{stage_id}_ngram_meta.json"),
    }
    if save_dense_matrices:
        paths["cooccurrence_matrix"] = str(matrices_dir / f"{stage_id}_cooccurrence.npy")
        paths["positional_dist"] = str(matrices_dir / f"{stage_id}_positional.npy")
    return paths, bigram_profile, trigram_profile, ttr, bg_cov, tg_cov, bg_ent, tg_ent


def _save_stage_corpus(
    stage_id: str,
    sequences: list[list[str]],
    corpora_dir: Path,
    preview_dir: Path,
    preview_count: int = PREVIEW_SEQUENCE_COUNT,
) -> dict[str, str]:
    """
    Save a generated stage corpus for later inspection.

    Returns a dict with paths to a full JSON dump and a plaintext preview.
    """
    corpora_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    corpus_path = corpora_dir / f"{stage_id}_tokens.json"
    preview_path = preview_dir / f"{stage_id}_preview.txt"

    with corpus_path.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "stage_id": stage_id,
                "num_sequences": len(sequences),
                "sequences": sequences,
            },
            fh,
            ensure_ascii=False,
            indent=2,
        )

    preview_lines = [" ".join(seq) for seq in sequences[:preview_count]]
    with preview_path.open("w", encoding="utf-8") as fh:
        fh.write("\n".join(preview_lines))
        if preview_lines:
            fh.write("\n")

    return {
        "corpus_json": str(corpus_path),
        "preview_txt": str(preview_path),
    }


# ---------------------------------------------------------------------------
# Option A: Stochastic engine (baby babble + reinforcement)
# ---------------------------------------------------------------------------

class StochasticRetrodictionEngine:
    """
    Random variation + Latin reinforcement.

    At each step: generate N random perturbations of the current bigram model,
    score each against Latin, keep the best. The baby makes random sounds;
    the ones that move toward Latin are reinforced.
    """

    def __init__(
        self,
        language: str,
        source_sequences: list[list[str]],
        latin_ref: LatinReference,
        config: ReinforcedConfig | None = None,
        output_dir: Path | None = None,
        references: ReferenceSet | None = None,
    ) -> None:
        self.language = language
        self.source_sequences = source_sequences
        self.latin_ref = latin_ref
        self.config = config or ReinforcedConfig()
        self.lang_code = LANG_CODES.get(language, language[:3].upper())
        self._references = references or ReferenceSet()

        if output_dir is None:
            output_dir = RETRODICTION_DIR / language / "stochastic"
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
        return f"{self.lang_code}_stoch_{iteration:03d}"

    def _perturb(
        self,
        model: BigramModel,
        rng: np.random.Generator,
    ) -> BigramModel:
        """
        Generate one random perturbation of the bigram model.
        Adds Gaussian noise in log-probability space, then re-normalises.
        """
        cfg = self.config
        log_T = np.log(model.transitions + 1e-10)
        log_T_noisy = log_T + cfg.noise_scale * rng.standard_normal(log_T.shape)
        # Softmax per row to get valid transition probabilities
        log_T_noisy -= log_T_noisy.max(axis=1, keepdims=True)
        T_new = np.exp(log_T_noisy)
        T_new /= T_new.sum(axis=1, keepdims=True)
        return BigramModel(model.vocab, model.token2idx, T_new, model.seq_lengths)

    def run(self) -> list[ReinforcedStageRecord]:
        cfg = self.config
        rng = np.random.default_rng(cfg.seed)

        log.info(
            "Stochastic retrodiction: language=%s, n_candidates=%d, noise=%.2f",
            self.language, cfg.n_candidates, cfg.noise_scale,
        )

        model = BigramModel.from_sequences(self.source_sequences)
        records: list[ReinforcedStageRecord] = []
        prev_score: float | None = None

        for iteration in range(cfg.max_iterations):
            stage_id = self._stage_id(iteration)

            # Generate candidates and pick the best
            best_model = None
            best_score = -np.inf
            best_vec = None
            best_seqs = None

            for _ in range(cfg.n_candidates):
                candidate = self._perturb(model, rng)
                seqs = candidate.sample_corpus(cfg.num_sequences, rng)
                # Quick score without full fingerprinting
                from src.ingest.tokenize import corpus_stats
                stats = corpus_stats(seqs)
                from src.fingerprint.ngram import extract_ngrams, build_profile
                bg_counts = extract_ngrams(seqs, 2)
                tg_counts = extract_ngrams(seqs, 3)
                bg_prof = build_profile(bg_counts, 5000)
                tg_prof = build_profile(tg_counts, 5000)
                vec = structural_vector(seqs, bg_prof, tg_prof)
                score = self.latin_ref.score(vec)
                if score > best_score:
                    best_score = score
                    best_model = candidate
                    best_vec = vec
                    best_seqs = seqs

            # Full fingerprint on winner
            fp_paths, bigram_profile, trigram_profile, ttr, bg_cov, tg_cov, bg_ent, tg_ent = (
                _fingerprint_sequences(stage_id, best_seqs, self.matrices_dir)
            )
            artifact_paths = _save_stage_corpus(
                stage_id,
                best_seqs,
                self.corpora_dir,
                self.preview_dir,
            )
            scores = self._references.score(best_seqs, bigram_profile, trigram_profile)
            diagnostics = self._references.coherence_from_vector(best_vec)

            record = ReinforcedStageRecord(
                stage_id=stage_id,
                source_language=self.language,
                algorithm="stochastic",
                iteration=iteration,
                fingerprint_paths=fp_paths,
                artifact_paths=artifact_paths,
                type_token_ratio=ttr,
                bigram_coverage=bg_cov,
                trigram_coverage=tg_cov,
                structural_vector=best_vec.tolist(),
                latin_score=best_score,
                scores=scores,
                bigram_entropy=bg_ent,
                trigram_entropy=tg_ent,
                diagnostics=diagnostics,
            )
            record.save(self.records_dir / f"{stage_id}.json")
            records.append(record)

            log.info(
                "Stoch %s: TTR=%.4f bg_cov=%.4f latin_score=%.4f coherence=%s",
                stage_id, ttr, bg_cov, best_score, diagnostics["coherence_label"],
            )

            # Stability check
            if prev_score is not None:
                delta = abs(best_score - prev_score)
                if delta < cfg.stability_threshold:
                    log.info("Stability at iteration %d (delta=%.6f). Halting.", iteration, delta)
                    record.flags.append("stable")
                    record.save(self.records_dir / f"{stage_id}.json")
                    break

            prev_score = best_score
            model = best_model

        self._save_summary(records, cfg)
        return records

    def _save_summary(self, records: list[ReinforcedStageRecord], cfg: ReinforcedConfig) -> None:
        best_record = max(records, key=lambda r: r.latin_score) if records else None
        summary = {
            "language": self.language,
            "algorithm": "stochastic",
            "latin_reward_features": list(REWARD_FEATURE_NAMES),
            "latin_reward_score_scale": REWARD_SCORE_SCALE,
            "config": cfg.to_dict(),
            "total_stages": len(records),
            "final_stage_id": records[-1].stage_id if records else None,
            "final_latin_score": records[-1].latin_score if records else None,
            "best_stage_id": best_record.stage_id if best_record else None,
            "best_latin_score": best_record.latin_score if best_record else None,
            "best_corpus_json": best_record.artifact_paths.get("corpus_json") if best_record else None,
            "best_preview_txt": best_record.artifact_paths.get("preview_txt") if best_record else None,
            "final_coherence_label": records[-1].diagnostics.get("coherence_label") if records else None,
            "final_language_likeness_margin": (
                records[-1].diagnostics.get("language_likeness_margin") if records else None
            ),
            "halted_stable": any("stable" in r.flags for r in records),
            "stages": [r.to_dict() for r in records],
        }
        path = self.output_dir / "run_summary.json"
        with path.open("w", encoding="utf-8") as fh:
            json.dump(summary, fh, ensure_ascii=False, indent=2)
        log.info("Saved stochastic summary to %s", path)


# ---------------------------------------------------------------------------
# Option B: Directed gradient engine
# ---------------------------------------------------------------------------

class GradientRetrodictionEngine:
    """
    Directed mixing toward Latin bigram model.

    At each step: mix the current transition matrix one step toward the Latin
    transition matrix. Deterministic. Finds the shortest path in
    transition-matrix space between the source language and Latin.
    """

    def __init__(
        self,
        language: str,
        source_sequences: list[list[str]],
        latin_ref: LatinReference,
        config: ReinforcedConfig | None = None,
        output_dir: Path | None = None,
        references: ReferenceSet | None = None,
    ) -> None:
        self.language = language
        self.source_sequences = source_sequences
        self.latin_ref = latin_ref
        self.config = config or ReinforcedConfig()
        self.lang_code = LANG_CODES.get(language, language[:3].upper())
        self._references = references or ReferenceSet()

        if output_dir is None:
            output_dir = RETRODICTION_DIR / language / "gradient"
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
        return f"{self.lang_code}_grad_{iteration:03d}"

    def _align_vocab(
        self,
        source_model: BigramModel,
        latin_model: BigramModel,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Align source and Latin transition matrices to a common vocabulary.
        Returns (source_T, latin_T) both shaped (V_common, V_common).
        Tokens absent from Latin get source's own transitions (no gradient).
        """
        # Common vocab: source vocab only (Latin vocab is large, we care about source tokens)
        V = source_model.V
        latin_T = np.zeros((V, V), dtype=np.float64)

        for i, tok_i in enumerate(source_model.vocab):
            if tok_i not in latin_model.token2idx:
                # Token not in Latin — keep source's distribution for this row
                latin_T[i] = source_model.transitions[i]
                continue
            li = latin_model.token2idx[tok_i]
            for j, tok_j in enumerate(source_model.vocab):
                if tok_j in latin_model.token2idx:
                    lj = latin_model.token2idx[tok_j]
                    latin_T[i, j] = latin_model.transitions[li, lj]
                # else: token not in Latin — leave 0

        # Renormalize rows (zero rows get uniform)
        row_sums = latin_T.sum(axis=1, keepdims=True)
        zero_rows = (row_sums.flatten() == 0)
        row_sums[zero_rows] = 1.0
        latin_T /= row_sums
        latin_T[zero_rows] = 1.0 / V

        return source_model.transitions, latin_T

    def run(self) -> list[ReinforcedStageRecord]:
        cfg = self.config
        rng = np.random.default_rng(cfg.seed)

        log.info(
            "Gradient retrodiction: language=%s, alpha=%.3f",
            self.language, cfg.alpha,
        )

        source_model = BigramModel.from_sequences(self.source_sequences)
        shared = sum(1 for tok in source_model.vocab if tok in self.latin_ref.model.token2idx)
        overlap_ratio = shared / source_model.V if source_model.V else 0.0
        if overlap_ratio < 0.10:
            log.warning(
                "Low source/Latin vocab overlap: %.2f%% shared tokens. "
                "Token-aligned gradient guidance is heuristic in this regime.",
                overlap_ratio * 100.0,
            )
        source_T, latin_T_aligned = self._align_vocab(source_model, self.latin_ref.model)

        current_T = source_T.copy()
        records: list[ReinforcedStageRecord] = []
        prev_score: float | None = None

        for iteration in range(cfg.max_iterations):
            stage_id = self._stage_id(iteration)

            # Build model from current transition matrix
            model = BigramModel(
                source_model.vocab,
                source_model.token2idx,
                current_T,
                source_model.seq_lengths,
            )

            # Generate and fingerprint
            seqs = model.sample_corpus(cfg.num_sequences, rng)
            fp_paths, bigram_profile, trigram_profile, ttr, bg_cov, tg_cov, bg_ent, tg_ent = (
                _fingerprint_sequences(stage_id, seqs, self.matrices_dir)
            )
            artifact_paths = _save_stage_corpus(
                stage_id,
                seqs,
                self.corpora_dir,
                self.preview_dir,
            )

            vec = structural_vector(seqs, bigram_profile, trigram_profile)
            latin_score = self.latin_ref.score(vec)
            scores = self._references.score(seqs, bigram_profile, trigram_profile)
            diagnostics = self._references.coherence_from_vector(vec)

            record = ReinforcedStageRecord(
                stage_id=stage_id,
                source_language=self.language,
                algorithm="gradient",
                iteration=iteration,
                fingerprint_paths=fp_paths,
                artifact_paths=artifact_paths,
                type_token_ratio=ttr,
                bigram_coverage=bg_cov,
                trigram_coverage=tg_cov,
                structural_vector=vec.tolist(),
                latin_score=latin_score,
                scores=scores,
                bigram_entropy=bg_ent,
                trigram_entropy=tg_ent,
                diagnostics=diagnostics,
            )
            record.save(self.records_dir / f"{stage_id}.json")
            records.append(record)

            log.info(
                "Grad %s: TTR=%.4f bg_cov=%.4f latin_score=%.4f coherence=%s",
                stage_id, ttr, bg_cov, latin_score, diagnostics["coherence_label"],
            )

            # Stability check
            if prev_score is not None:
                delta = abs(latin_score - prev_score)
                if delta < cfg.stability_threshold:
                    log.info("Stability at iteration %d (delta=%.6f). Halting.", iteration, delta)
                    record.flags.append("stable")
                    record.save(self.records_dir / f"{stage_id}.json")
                    break

            prev_score = latin_score

            # Mix one step toward Latin
            current_T = (1.0 - cfg.alpha) * current_T + cfg.alpha * latin_T_aligned

        self._save_summary(records, cfg, overlap_ratio)
        return records

    def _save_summary(
        self,
        records: list[ReinforcedStageRecord],
        cfg: ReinforcedConfig,
        overlap_ratio: float,
    ) -> None:
        best_record = max(records, key=lambda r: r.latin_score) if records else None
        summary = {
            "language": self.language,
            "algorithm": "gradient",
            "latin_reward_features": list(REWARD_FEATURE_NAMES),
            "latin_reward_score_scale": REWARD_SCORE_SCALE,
            "latin_vocab_overlap_ratio": overlap_ratio,
            "config": cfg.to_dict(),
            "total_stages": len(records),
            "final_stage_id": records[-1].stage_id if records else None,
            "final_latin_score": records[-1].latin_score if records else None,
            "best_stage_id": best_record.stage_id if best_record else None,
            "best_latin_score": best_record.latin_score if best_record else None,
            "best_corpus_json": best_record.artifact_paths.get("corpus_json") if best_record else None,
            "best_preview_txt": best_record.artifact_paths.get("preview_txt") if best_record else None,
            "final_coherence_label": records[-1].diagnostics.get("coherence_label") if records else None,
            "final_language_likeness_margin": (
                records[-1].diagnostics.get("language_likeness_margin") if records else None
            ),
            "halted_stable": any("stable" in r.flags for r in records),
            "stages": [r.to_dict() for r in records],
        }
        path = self.output_dir / "run_summary.json"
        with path.open("w", encoding="utf-8") as fh:
            json.dump(summary, fh, ensure_ascii=False, indent=2)
        log.info("Saved gradient summary to %s", path)


# ---------------------------------------------------------------------------
# Convenience runner
# ---------------------------------------------------------------------------

def run(
    language: str,
    algorithm: str = "both",
    config: ReinforcedConfig | None = None,
    input_path: Path | None = None,
) -> dict[str, list[ReinforcedStageRecord]]:
    """
    Run reinforcement retrodiction for a single language.

    Args:
        language:   Language name.
        algorithm:  "stochastic", "gradient", or "both".
        config:     ReinforcedConfig (uses defaults if None).
        input_path: Path to tokenized corpus JSON.

    Returns:
        Dict with keys "stochastic" and/or "gradient" mapping to record lists.
    """
    if input_path is None:
        input_path = PROCESSED_DIR / "romance" / f"{language}_tokens.json"

    log.info("Loading source corpus from %s", input_path)
    with input_path.open(encoding="utf-8") as fh:
        corpus = json.load(fh)
    sequences = corpus["sequences"]
    log.info("Loaded %d sequences", len(sequences))

    cfg = config or ReinforcedConfig()
    latin_ref = LatinReference()   # loads Latin once, re-locks sequestration
    references = ReferenceSet()

    results = {}

    if algorithm in ("stochastic", "both"):
        engine = StochasticRetrodictionEngine(language, sequences, latin_ref, cfg, references=references)
        results["stochastic"] = engine.run()

    if algorithm in ("gradient", "both"):
        engine = GradientRetrodictionEngine(language, sequences, latin_ref, cfg, references=references)
        results["gradient"] = engine.run()

    return results
