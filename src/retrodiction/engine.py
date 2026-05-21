"""
Retrodiction Engine
===================
Purpose:
    Run a single-language retrodiction: starting from a modern Romance corpus,
    iteratively apply backward transformations, generate synthetic intermediate
    corpora, fingerprint each stage, score against null models, and record
    every bridge stage.

Algorithm:
    1. Build a BigramModel from the source corpus.
    2. Sample a synthetic corpus from the model.
    3. Fingerprint the synthetic corpus (cooccurrence, positional, ngram).
    4. Score the fingerprint against Markov noise and Sumerian references.
    5. Store the complete bridge stage record.
    6. Mix the model toward uniform (alpha step).
    7. Check stability: if the structural vector has not changed meaningfully,
       halt. Otherwise repeat from 2.

Halting:
    The algorithm halts when the L2 distance between consecutive structural
    vectors falls below STABILITY_THRESHOLD, or after MAX_ITERATIONS steps.
    The stable point is the finding — it is not tuned toward Latin.

Output structure:
    data/retrodiction/{language}/
        records/
            {LANG}_retro_{ITER:03d}.json   # bridge stage record
        matrices/
            {LANG}_retro_{ITER:03d}_cooccurrence.npy
            {LANG}_retro_{ITER:03d}_positional.npy
            {LANG}_retro_{ITER:03d}_ngram_meta.json

Bridge stage record format:
    {
      "stage_id": "FR_retro_014",
      "source_language": "french",
      "iteration": 14,
      "alpha_cumulative": 0.54,
      "fingerprint": {
        "cooccurrence_matrix": "data/retrodiction/french/matrices/FR_retro_014_cooccurrence.npy",
        "positional_dist":     "data/retrodiction/french/matrices/FR_retro_014_positional.npy",
        "ngram_meta":          "data/retrodiction/french/matrices/FR_retro_014_ngram_meta.json",
        "type_token_ratio":    0.1291,
        "bigram_entropy":      4.312,
        "trigram_entropy":     5.891
      },
      "scores": {
        "vs_markov_noise":       0.743,
        "vs_sumerian":           0.612,
        "vs_portuguese_control": null,
        "vs_latin_ground_truth": null
      },
      "structural_vector": [0.1291, 4.312, 5.891, 2.303],
      "notes": "",
      "flags": []
    }
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.retrodiction.generate import BigramModel
from src.retrodiction.similarity import ReferenceSet, structural_vector, top_k_coverage
from src.fingerprint import cooccurrence, positional, ngram

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RETRODICTION_DIR = PROJECT_ROOT / "data" / "retrodiction"

# Language code abbreviations for stage IDs
LANG_CODES = {
    "french": "FR",
    "italian": "IT",
    "spanish": "ES",
    "romanian": "RO",
    "occitan": "OC",
    "genoese": "LIJ",
}

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class RetrodictionConfig:
    """All tunable parameters for a retrodiction run."""
    alpha: float = 0.05             # mixing rate toward uniform per step
    num_sequences: int = 2000       # synthetic sequences per stage
    max_iterations: int = 200       # hard upper limit
    stability_threshold: float = 0.005  # L2 distance to declare stability
    seed: int = 42                  # random seed for reproducibility

    def to_dict(self) -> dict:
        return {
            "alpha": self.alpha,
            "num_sequences": self.num_sequences,
            "max_iterations": self.max_iterations,
            "stability_threshold": self.stability_threshold,
            "seed": self.seed,
        }


# ---------------------------------------------------------------------------
# Bridge stage record
# ---------------------------------------------------------------------------

@dataclass
class BridgeStageRecord:
    stage_id: str
    source_language: str
    iteration: int
    alpha_cumulative: float
    fingerprint_paths: dict
    type_token_ratio: float
    bigram_coverage: float    # top-100 bigram coverage (primary discriminator)
    trigram_coverage: float   # top-100 trigram coverage
    bigram_entropy: float     # retained for inspection only, not used in scoring
    trigram_entropy: float    # retained for inspection only, not used in scoring
    scores: dict
    structural_vector: list[float]
    notes: str = ""
    flags: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "stage_id": self.stage_id,
            "source_language": self.source_language,
            "iteration": self.iteration,
            "alpha_cumulative": round(self.alpha_cumulative, 6),
            "fingerprint": {
                **self.fingerprint_paths,
                "type_token_ratio": self.type_token_ratio,
                "bigram_coverage": self.bigram_coverage,
                "trigram_coverage": self.trigram_coverage,
                "bigram_entropy": self.bigram_entropy,
                "trigram_entropy": self.trigram_entropy,
            },
            "scores": self.scores,
            "structural_vector": [round(v, 6) for v in self.structural_vector],
            "notes": self.notes,
            "flags": self.flags,
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class RetrodictionEngine:
    """
    Runs a single-language retrodiction from a modern corpus backward through
    bridge stages, recording every intermediate.
    """

    def __init__(
        self,
        language: str,
        source_sequences: list[list[str]],
        config: RetrodictionConfig | None = None,
        output_dir: Path | None = None,
    ) -> None:
        self.language = language
        self.source_sequences = source_sequences
        self.config = config or RetrodictionConfig()
        self.lang_code = LANG_CODES.get(language, language[:3].upper())

        if output_dir is None:
            output_dir = RETRODICTION_DIR / language
        self.output_dir = output_dir
        self.records_dir = output_dir / "records"
        self.matrices_dir = output_dir / "matrices"

        self.records_dir.mkdir(parents=True, exist_ok=True)
        self.matrices_dir.mkdir(parents=True, exist_ok=True)

        self._references = ReferenceSet()

    def _stage_id(self, iteration: int) -> str:
        return f"{self.lang_code}_retro_{iteration:03d}"

    def _run_fingerprint(
        self,
        stage_id: str,
        sequences: list[list[str]],
    ) -> tuple[dict, dict, dict, float, float, float, float, float]:
        """
        Fingerprint a generated corpus and save matrices.

        Returns:
            (fingerprint_paths, bigram_profile, trigram_profile,
             ttr, bg_coverage, tg_coverage, bg_ent, tg_ent)
        """
        import math

        cooc_matrix, _, _ = cooccurrence.run_from_sequences(
            stage_id, sequences, output_dir=self.matrices_dir
        )
        positional.run_from_sequences(
            stage_id, sequences, output_dir=self.matrices_dir
        )
        bigram_profile, trigram_profile, ttr = ngram.run_from_sequences(
            stage_id, sequences, output_dir=self.matrices_dir
        )

        bg_coverage = top_k_coverage(bigram_profile)
        tg_coverage = top_k_coverage(trigram_profile)
        bg_ent = sum(-p * math.log(p) for p in bigram_profile.values() if p > 0)
        tg_ent = sum(-p * math.log(p) for p in trigram_profile.values() if p > 0)

        paths = {
            "cooccurrence_matrix": str(self.matrices_dir / f"{stage_id}_cooccurrence.npy"),
            "positional_dist":     str(self.matrices_dir / f"{stage_id}_positional.npy"),
            "ngram_meta":          str(self.matrices_dir / f"{stage_id}_ngram_meta.json"),
        }
        return paths, bigram_profile, trigram_profile, ttr, bg_coverage, tg_coverage, bg_ent, tg_ent

    def run(self) -> list[BridgeStageRecord]:
        """
        Execute the full retrodiction loop.

        Returns:
            List of BridgeStageRecord, one per iteration.
        """
        cfg = self.config
        rng = np.random.default_rng(cfg.seed)

        log.info(
            "Starting retrodiction: language=%s, alpha=%.3f, max_iter=%d",
            self.language, cfg.alpha, cfg.max_iterations,
        )

        model = BigramModel.from_sequences(self.source_sequences)
        records: list[BridgeStageRecord] = []
        prev_vec: np.ndarray | None = None
        alpha_cumulative = 0.0

        for iteration in range(cfg.max_iterations):
            stage_id = self._stage_id(iteration)
            log.info("--- Stage %s (alpha_cumulative=%.3f) ---", stage_id, alpha_cumulative)

            # Generate synthetic corpus from current model
            sequences = model.sample_corpus(cfg.num_sequences, rng)

            # Fingerprint the generated corpus
            fp_paths, bigram_profile, trigram_profile, ttr, bg_cov, tg_cov, bg_ent, tg_ent = (
                self._run_fingerprint(stage_id, sequences)
            )

            # Score against references
            scores = self._references.score(sequences, bigram_profile, trigram_profile)

            # Structural vector for stability check
            svec = structural_vector(sequences, bigram_profile, trigram_profile)

            record = BridgeStageRecord(
                stage_id=stage_id,
                source_language=self.language,
                iteration=iteration,
                alpha_cumulative=alpha_cumulative,
                fingerprint_paths=fp_paths,
                type_token_ratio=ttr,
                bigram_coverage=bg_cov,
                trigram_coverage=tg_cov,
                bigram_entropy=bg_ent,
                trigram_entropy=tg_ent,
                scores=scores,
                structural_vector=svec.tolist(),
            )
            record.save(self.records_dir / f"{stage_id}.json")
            records.append(record)

            log.info(
                "Stage %s: TTR=%.4f, bg_cov=%.4f, vs_markov=%.4f, vs_sumerian=%.4f",
                stage_id, ttr, bg_cov,
                scores["vs_markov_noise"], scores["vs_sumerian"],
            )

            # Stability check
            if prev_vec is not None:
                delta = float(np.linalg.norm(svec - prev_vec))
                log.info("Structural vector delta: %.6f (threshold=%.6f)", delta, cfg.stability_threshold)
                if delta < cfg.stability_threshold:
                    log.info("Stability reached at iteration %d. Halting.", iteration)
                    record.flags.append("stable")
                    record.save(self.records_dir / f"{stage_id}.json")
                    break

            prev_vec = svec

            # Apply backward transformation
            model = model.mix_toward_uniform(cfg.alpha)
            alpha_cumulative += cfg.alpha * (1.0 - alpha_cumulative)  # effective cumulative mixing

        log.info(
            "Retrodiction complete: %d stages, language=%s",
            len(records), self.language,
        )
        self._save_run_summary(records, cfg)
        return records

    def _save_run_summary(
        self,
        records: list[BridgeStageRecord],
        cfg: RetrodictionConfig,
    ) -> None:
        summary = {
            "language": self.language,
            "lang_code": self.lang_code,
            "config": cfg.to_dict(),
            "total_stages": len(records),
            "final_stage_id": records[-1].stage_id if records else None,
            "halted_stable": any("stable" in r.flags for r in records),
            "stages": [r.to_dict() for r in records],
        }
        summary_path = self.output_dir / "run_summary.json"
        with summary_path.open("w", encoding="utf-8") as fh:
            json.dump(summary, fh, ensure_ascii=False, indent=2)
        log.info("Saved run summary to %s", summary_path)


# ---------------------------------------------------------------------------
# Convenience entry point
# ---------------------------------------------------------------------------

def run(
    language: str,
    config: RetrodictionConfig | None = None,
    input_path: Path | None = None,
) -> list[BridgeStageRecord]:
    """
    Run retrodiction for a single language.

    Args:
        language:   Language name (must be in LANG_CODES).
        config:     RetrodictionConfig (uses defaults if None).
        input_path: Path to the tokenized corpus JSON. Defaults to standard location.

    Returns:
        List of BridgeStageRecord for all stages.
    """
    if input_path is None:
        input_path = (
            PROJECT_ROOT / "data" / "processed" / "romance" / f"{language}_tokens.json"
        )

    log.info("Loading source corpus from %s", input_path)
    with input_path.open(encoding="utf-8") as fh:
        corpus = json.load(fh)
    sequences = corpus["sequences"]
    log.info("Loaded %d sequences", len(sequences))

    engine = RetrodictionEngine(language, sequences, config)
    return engine.run()
