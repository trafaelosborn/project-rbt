"""
V5 vs V4 Head-to-Head Probe
=============================
Usage:
    python -m src.accelerate.v5_vs_v4_headtohead

Runs matched short probes:
  - v4 (baseline): incremental scoring, Python candidate scoring
  - v5 (plain): same search logic as v4, no culture bombs, Python scoring
  - v5f (Fortran batch): same as plain v5, batched form scoring enabled

All three start from the same source corpus, same seed family.
Reports struct/form/alignment trajectories and proposals/hour for each.
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from pathlib import Path

from src.retrodiction.engine_reinforced import LatinReference
from src.retrodiction.engine_reinforced_v2 import LatinFormReference
from src.retrodiction.engine_reinforced_v4 import (
    ReinforcedV4Config,
    RelationalReinforcedRetrodictionEngineV4,
)
from src.retrodiction.engine_reinforced_v5 import (
    ReinforcedV5Config,
    RelationalReinforcedRetrodictionEngineV5,
)
from src.retrodiction.similarity import ReferenceSet
from src.validation.hungarian_alignment import extract_family_inventory

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SOURCE_CORPUS = PROJECT_ROOT / "data/processed/romance/french_tokens.json"
PROBE_PROPOSALS = 100
PROBE_SEED = 77
PROBE_CANDIDATES = 16


def _load_corpus(path: Path) -> list[list[str]]:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)["sequences"]


def _load_latin_sample() -> list[list[str]]:
    latin_path = PROJECT_ROOT / "data/sequestered/latin/latin_tokens.json"
    with latin_path.open(encoding="utf-8") as fh:
        return json.load(fh)["sequences"][:72]


def _run_v4_probe(sequences, output_dir, latin_structural_ref, latin_form_ref, references) -> dict:
    latin_seqs = _load_latin_sample()
    cfg = ReinforcedV4Config(
        num_sequences=800,
        max_proposals=PROBE_PROPOSALS,
        max_accepted_stages=512,
        patience=PROBE_PROPOSALS,
        seed=PROBE_SEED,
        n_candidates=PROBE_CANDIDATES,
        min_improvement=0.0001,
        save_dense_matrices=False,
        use_incremental_scoring=True,
        use_semantic_transparency=False,
        use_fortran_batch=False,
    )
    family_ref = extract_family_inventory("latin", latin_seqs, cfg.alignment_config)
    engine = RelationalReinforcedRetrodictionEngineV4(
        language="french",
        source_sequences=list(sequences),
        latin_structural_ref=latin_structural_ref,
        latin_form_ref=latin_form_ref,
        config=cfg,
        output_dir=output_dir,
        references=references,
        family_reference_inventory=family_ref,
    )
    t0 = time.perf_counter()
    engine.run()
    elapsed = time.perf_counter() - t0
    return _read_summary(output_dir, elapsed, "v4_baseline")


def _run_v5_probe(sequences, output_dir, latin_structural_ref, latin_form_ref, references) -> dict:
    """V5 plain: no culture bombs, Python candidate scoring."""
    latin_seqs = _load_latin_sample()
    cfg = ReinforcedV5Config(
        num_sequences=800,
        max_proposals=PROBE_PROPOSALS,
        max_accepted_stages=512,
        patience=PROBE_PROPOSALS,
        seed=PROBE_SEED,
        n_candidates=PROBE_CANDIDATES,
        min_improvement=0.0001,
        save_dense_matrices=False,
        use_incremental_scoring=True,
        use_fortran_batch=False,
        use_semantic_transparency=False,
        enable_culture_bombs=False,
    )
    family_ref = extract_family_inventory("latin", latin_seqs, cfg.alignment_config)
    engine = RelationalReinforcedRetrodictionEngineV5(
        language="french",
        source_sequences=list(sequences),
        latin_structural_ref=latin_structural_ref,
        latin_form_ref=latin_form_ref,
        config=cfg,
        output_dir=output_dir,
        references=references,
        family_reference_inventory=family_ref,
    )
    t0 = time.perf_counter()
    engine.run()
    elapsed = time.perf_counter() - t0
    return _read_summary(output_dir, elapsed, "v5_plain_python")


def _run_v5f_probe(sequences, output_dir, latin_structural_ref, latin_form_ref, references) -> dict:
    """V5f: no culture bombs, batched form scoring enabled."""
    latin_seqs = _load_latin_sample()
    cfg = ReinforcedV5Config(
        num_sequences=800,
        max_proposals=PROBE_PROPOSALS,
        max_accepted_stages=512,
        patience=PROBE_PROPOSALS,
        seed=PROBE_SEED,
        n_candidates=PROBE_CANDIDATES,
        min_improvement=0.0001,
        save_dense_matrices=False,
        use_incremental_scoring=True,
        use_fortran_cosine=True,
        use_fortran_batch=True,
        use_semantic_transparency=False,
        enable_culture_bombs=False,
    )
    family_ref = extract_family_inventory("latin", latin_seqs, cfg.alignment_config)
    engine = RelationalReinforcedRetrodictionEngineV5(
        language="french",
        source_sequences=list(sequences),
        latin_structural_ref=latin_structural_ref,
        latin_form_ref=latin_form_ref,
        config=cfg,
        output_dir=output_dir,
        references=references,
        family_reference_inventory=family_ref,
    )
    t0 = time.perf_counter()
    engine.run()
    elapsed = time.perf_counter() - t0
    return _read_summary(output_dir, elapsed, "v5_fortran_batch")


def _read_summary(output_dir: Path, elapsed: float, label: str) -> dict:
    with (output_dir / "run_summary.json").open(encoding="utf-8") as fh:
        summary = json.load(fh)
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
        "halt_reason": summary.get("halt_reason"),
    }


def run_headtohead(output_dir: Path | None = None) -> dict:
    if output_dir is None:
        output_dir = PROJECT_ROOT / "data/benchmarks/v5_vs_v4_headtohead"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not SOURCE_CORPUS.exists():
        raise FileNotFoundError(f"Source corpus not found: {SOURCE_CORPUS}")

    sequences = _load_corpus(SOURCE_CORPUS)
    log.info("Loaded %d sequences from source corpus", len(sequences))

    latin_structural_ref = LatinReference()
    latin_form_ref = LatinFormReference()
    references = ReferenceSet()

    results = []
    for label, runner in [
        ("v4_baseline", _run_v4_probe),
        ("v5_plain_python", _run_v5_probe),
        ("v5_fortran_batch", _run_v5f_probe),
    ]:
        probe_dir = output_dir / label
        shutil.rmtree(probe_dir, ignore_errors=True)
        probe_dir.mkdir(parents=True, exist_ok=True)
        log.info("Running probe: %s (%d proposals)", label, PROBE_PROPOSALS)
        result = runner(sequences, probe_dir, latin_structural_ref, latin_form_ref, references)
        results.append(result)
        log.info(
            "  %-30s struct=%.4f form=%.4f accepted=%d throughput=%.0f p/h halt=%s",
            label,
            result["final_struct"] or 0,
            result["final_form"] or 0,
            result["accepted_stages"],
            result["proposals_per_hour"],
            result["halt_reason"],
        )

    report = {
        "probe_proposals": PROBE_PROPOSALS,
        "probe_seed": PROBE_SEED,
        "probe_candidates": PROBE_CANDIDATES,
        "start_corpus": str(SOURCE_CORPUS),
        "results": results,
    }
    out_path = output_dir / "headtohead_report.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    log.info("Head-to-head report written to %s", out_path)

    return report


if __name__ == "__main__":
    result = run_headtohead()
    for r in result["results"]:
        print(
            f"{r['label']:<35} struct={r['final_struct']:+.4f}  "
            f"form={r['final_form']:.4f}  "
            f"accepted={r['accepted_stages']}  "
            f"{r['proposals_per_hour']:,.0f} p/h"
        )
