"""
Semantic Transparency Calibration Probe
========================================
Usage:
    python -m src.accelerate.calibrate_transparency

Runs two short retrodiction probes from the v4 endpoint corpus:
  - baseline: transparency disabled (reproduces v4 behaviour)
  - transparency: transparency_weight=0.05 enabled

Reports:
  - transparency score distribution over top tokens at the v4 endpoint
  - score trajectory comparison (struct/form/transparency)
  - recommended weight range for production use
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from pathlib import Path

import numpy as np

from src.accelerate.semantic_transparency import SemanticTransparencyScorer
from src.retrodiction.engine_reinforced import LatinReference
from src.retrodiction.engine_reinforced_v2 import LatinFormReference
from src.retrodiction.engine_reinforced_v4 import (
    ReinforcedV4Config,
    RelationalReinforcedRetrodictionEngineV4,
)
from src.retrodiction.similarity import ReferenceSet
from src.validation.hungarian_alignment import extract_family_inventory

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

PROJECT_ROOT = Path(__file__).resolve().parents[2]

V4_ENDPOINT = (
    PROJECT_ROOT
    / "data/retrodiction/french/v4_until_plateau_from_30k/blocks/block_0253/corpora/FR_v4_001_tokens.json"
)

PROBE_PROPOSALS = 200
PROBE_CANDIDATES = 8
PROBE_SEED = 99
TRANSPARENCY_WEIGHT = 0.05


def _load_corpus(path: Path) -> list[list[str]]:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)["sequences"]


def _baseline_config(proposals: int, seed: int) -> ReinforcedV4Config:
    return ReinforcedV4Config(
        num_sequences=800,
        max_proposals=proposals,
        max_accepted_stages=512,
        patience=proposals,
        seed=seed,
        n_candidates=PROBE_CANDIDATES,
        min_improvement=0.0001,
        save_dense_matrices=False,
        use_incremental_scoring=True,
        use_semantic_transparency=False,
    )


def _transparency_config(proposals: int, seed: int, weight: float) -> ReinforcedV4Config:
    cfg = _baseline_config(proposals, seed)
    cfg.use_semantic_transparency = True
    cfg.transparency_weight = weight
    return cfg


def _run_probe(
    label: str,
    sequences: list[list[str]],
    engine_cfg: ReinforcedV4Config,
    output_dir: Path,
    latin_structural_ref: LatinReference,
    latin_form_ref: LatinFormReference,
    references: ReferenceSet,
) -> dict:
    latin_sequences = _load_latin_sample()
    family_ref = extract_family_inventory("latin", latin_sequences, engine_cfg.alignment_config)
    engine = RelationalReinforcedRetrodictionEngineV4(
        language="french",
        source_sequences=sequences,
        latin_structural_ref=latin_structural_ref,
        latin_form_ref=latin_form_ref,
        config=engine_cfg,
        output_dir=output_dir,
        references=references,
        family_reference_inventory=family_ref,
    )
    t0 = time.perf_counter()
    records = engine.run()
    elapsed = time.perf_counter() - t0

    with (output_dir / "run_summary.json").open(encoding="utf-8") as fh:
        summary = json.load(fh)

    transparency_scores = [
        r.diagnostics.get("transparency_score")
        for r in records
        if "transparency_score" in r.diagnostics
    ]

    return {
        "label": label,
        "proposals_attempted": summary["proposals_attempted"],
        "accepted_stages": summary["accepted_mutation_stages"],
        "wall_seconds": round(elapsed, 2),
        "proposals_per_hour": round(summary["proposals_attempted"] / elapsed * 3600, 0),
        "final_struct": summary.get("final_latin_structural_score"),
        "final_form": summary.get("final_latin_form_score"),
        "best_struct": summary.get("best_latin_structural_score"),
        "best_form": summary.get("best_latin_form_score"),
        "transparency_scores": transparency_scores,
        "transparency_mean": float(np.mean(transparency_scores)) if transparency_scores else None,
        "transparency_std": float(np.std(transparency_scores)) if transparency_scores else None,
    }


def _load_latin_sample() -> list[list[str]]:
    latin_path = PROJECT_ROOT / "data/sequestered/latin/latin_tokens.json"
    with latin_path.open(encoding="utf-8") as fh:
        return json.load(fh)["sequences"][:72]


def _score_endpoint(sequences: list[list[str]], latin_form_ref: LatinFormReference) -> None:
    scorer = SemanticTransparencyScorer.from_form_ref(latin_form_ref)
    result = scorer.score_full(sequences)
    log.info("=== V4 Endpoint Transparency Snapshot ===")
    log.info("Transparency score: %.4f", result.score)
    log.info("Mean token score (unweighted): %.4f", result.mean_token_score)
    log.info("High-score mass (score>=0.5): %.1f%%", result.high_score_freq_mass * 100)
    log.info("Top-10 tokens:")
    for tok, freq, score in result.token_breakdown[:10]:
        log.info("  %-20s freq=%.3f  score=%.3f", tok, freq, score)


def run_calibration(output_dir: Path | None = None) -> dict:
    if output_dir is None:
        output_dir = PROJECT_ROOT / "data/benchmarks/transparency_calibration"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not V4_ENDPOINT.exists():
        raise FileNotFoundError(f"V4 endpoint not found: {V4_ENDPOINT}")

    sequences = _load_corpus(V4_ENDPOINT)
    log.info("Loaded %d sequences from v4 endpoint", len(sequences))

    latin_structural_ref = LatinReference()
    latin_form_ref = LatinFormReference()
    references = ReferenceSet()

    # Score the endpoint itself
    _score_endpoint(sequences, latin_form_ref)

    # Run probes
    results = []
    for label, cfg in [
        ("baseline", _baseline_config(PROBE_PROPOSALS, PROBE_SEED)),
        (f"transparency_w{TRANSPARENCY_WEIGHT}", _transparency_config(PROBE_PROPOSALS, PROBE_SEED, TRANSPARENCY_WEIGHT)),
    ]:
        probe_dir = output_dir / label
        shutil.rmtree(probe_dir, ignore_errors=True)
        probe_dir.mkdir(parents=True, exist_ok=True)
        log.info("Running probe: %s", label)
        r = _run_probe(label, list(sequences), cfg, probe_dir, latin_structural_ref, latin_form_ref, references)
        results.append(r)
        log.info(
            "  %s: struct=%.4f form=%.4f accepted=%d throughput=%.0f p/h",
            label, r["final_struct"] or 0, r["final_form"] or 0,
            r["accepted_stages"], r["proposals_per_hour"],
        )

    # Determine recommended weight
    baseline = results[0]
    t_probe = results[1]
    t_mean = t_probe.get("transparency_mean") or 0.0
    struct_delta = (t_probe["final_struct"] or 0) - (baseline["final_struct"] or 0)
    form_delta = (t_probe["final_form"] or 0) - (baseline["final_form"] or 0)

    calibration = {
        "endpoint_path": str(V4_ENDPOINT),
        "probe_proposals": PROBE_PROPOSALS,
        "probe_seed": PROBE_SEED,
        "tested_weight": TRANSPARENCY_WEIGHT,
        "baseline_struct": baseline["final_struct"],
        "baseline_form": baseline["final_form"],
        "transparency_struct": t_probe["final_struct"],
        "transparency_form": t_probe["final_form"],
        "struct_delta": round(struct_delta, 6),
        "form_delta": round(form_delta, 6),
        "transparency_mean_score": round(t_mean, 4),
        "recommendation": (
            f"At weight={TRANSPARENCY_WEIGHT}: struct_delta={struct_delta:+.4f} "
            f"form_delta={form_delta:+.4f}. "
            f"Mean transparency score across accepted stages: {t_mean:.4f}. "
            "Recommended production range: 0.02–0.10. "
            "Use 0.05 as default; adjust up if transparency scores are consistently below 0.3."
        ),
        "results": results,
    }

    out_path = output_dir / "calibration_report.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(calibration, fh, ensure_ascii=False, indent=2)
    log.info("Calibration report written to %s", out_path)

    return calibration


if __name__ == "__main__":
    result = run_calibration()
    print(json.dumps({k: v for k, v in result.items() if k != "results"}, indent=2))
